from __future__ import annotations

from fractions import Fraction

import pytest

from oamb.contracts.ports import FinishDisposition, ModelCandidate, ModelCompletion
from oamb.workloads.redial import (
    MovieCatalog,
    parse_ranked_movies,
    resolve_ranked_movies,
    score_recall_at_5,
)


def _completion(
    text: str, *, finish: FinishDisposition = FinishDisposition.NORMAL_STOP
) -> ModelCompletion:
    return ModelCompletion(
        finish_disposition=finish,
        candidates=(ModelCandidate(content=text, tool_call_present=False, complete=True),),
    )


def test_redial_ranked_parser_accepts_header_crlf_short_lists_and_duplicate_titles() -> None:
    parsed = parse_ranked_movies(
        _completion("The Recommendations Are:\r\n1. Alien (1979)\r\n\r\n2、Alien (1979)\r\n")
    )

    assert parsed == ("Alien (1979)", "Alien (1979)")
    for invalid in (
        "Alien, Aliens",
        "Here are movies:\n1. Alien",
        "1. Alien\n3. Arrival",
        "2. Alien",
        "1. Alien\n1. Arrival",
        "21. Alien",
    ):
        with pytest.raises(ValueError):
            parse_ranked_movies(_completion(invalid))
    with pytest.raises(ValueError, match="normal stop"):
        parse_ranked_movies(_completion("1. Alien", finish=FinishDisposition.LENGTH_LIMIT))


def test_redial_resolution_ties_and_colliding_gold_occurrences_are_rebuildable() -> None:
    catalog = MovieCatalog.from_entity_map(
        {
            "movie/Alien_(1979)": 20,
            "movie/ALIEN_(1979)": 10,
            "movie/Arrival_(2016)": 30,
        }
    )

    resolved = resolve_ranked_movies(("alien", "Arrival (2016)", "alien"), catalog)
    score = score_recall_at_5(resolved, (10, 20, 30), catalog)

    assert resolved[0].entity_id == 10
    assert resolved[0].tied_entity_ids == (10, 20)
    assert score.value == Fraction(3, 3)
    assert score.numerator == 3
    assert score.denominator == 3


def test_redial_catalog_rejects_non_integer_or_duplicate_entity_ids() -> None:
    with pytest.raises(ValueError, match="entity IDs"):
        MovieCatalog.from_entity_map({"movie/Alien": 1, "movie/Arrival": 1})
    with pytest.raises(ValueError, match="entity IDs"):
        MovieCatalog.from_entity_map({"movie/Alien": True})
