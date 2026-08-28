"""Strict ReDial ranked output, deterministic catalog resolution, and Recall@5."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from urllib.parse import unquote, urlsplit

from oamb.contracts.ports import FinishDisposition, ModelCompletion

_HEADER = "the recommendations are:"
_RANKED_LINE = re.compile(r"^[ \t]*(?:[1-9]|1[0-9]|20)[.)、][ \t]*(?P<title>\S.*?)[ \t]*$")
_RANK = re.compile(r"^[ \t]*(?P<rank>[1-9]|1[0-9]|20)[.)、]")
_PARENTHETICAL = re.compile(r"\([^()]*\)")


@dataclass(frozen=True, slots=True)
class CatalogMovie:
    entity_id: int
    source_uri: str
    cleaned_title: str
    normalized_title: str


@dataclass(frozen=True, slots=True)
class MovieCatalog:
    movies: tuple[CatalogMovie, ...]
    unicode_version: str
    duplicate_normalized_titles: tuple[tuple[str, tuple[int, ...]], ...]

    @classmethod
    def from_entity_map(cls, entity_map: Mapping[str, int]) -> MovieCatalog:
        entity_ids = tuple(entity_map.values())
        if (
            not entity_ids
            or any(type(entity_id) is not int or entity_id < 0 for entity_id in entity_ids)
            or len(set(entity_ids)) != len(entity_ids)
        ):
            raise ValueError("ReDial catalog entity IDs must be unique non-negative integers")
        movies = tuple(
            CatalogMovie(
                entity_id=int(entity_id),
                source_uri=uri,
                cleaned_title=clean_catalog_title(uri),
                normalized_title=normalize_movie_title(clean_catalog_title(uri)),
            )
            for uri, entity_id in entity_map.items()
        )
        if any(not movie.cleaned_title for movie in movies):
            raise ValueError("ReDial catalog requires non-empty cleaned movie titles")
        grouped: dict[str, list[int]] = {}
        for movie in movies:
            grouped.setdefault(movie.normalized_title, []).append(movie.entity_id)
        duplicates = tuple(
            (title, tuple(sorted(ids)))
            for title, ids in sorted(grouped.items(), key=lambda item: item[0].encode())
            if len(ids) > 1
        )
        return cls(
            movies=movies,
            unicode_version=unicodedata.unidata_version,
            duplicate_normalized_titles=duplicates,
        )


@dataclass(frozen=True, slots=True)
class ResolvedMovie:
    raw_title: str
    cleaned_title: str
    normalized_title: str
    entity_id: int
    catalog_title: str
    catalog_normalized_title: str
    distance: int
    tied_entity_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RecallAtFiveResult:
    numerator: int
    denominator: int
    value: Fraction
    gold_entity_ids: tuple[int, ...]
    gold_normalized_titles: tuple[str, ...]
    predicted_normalized_titles: tuple[str, ...]


def parse_ranked_movies(completion: ModelCompletion) -> tuple[str, ...]:
    if completion.finish_disposition != FinishDisposition.NORMAL_STOP:
        raise ValueError("ReDial output requires a normal stop")
    if len(completion.candidates) != 1:
        raise ValueError("ReDial output requires exactly one candidate")
    candidate = completion.candidates[0]
    if candidate.content is None or candidate.tool_call_present or not candidate.complete:
        raise ValueError("ReDial candidate must be complete text without a tool call")
    normalized = candidate.content.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line for line in normalized.split("\n") if line.strip()]
    if lines and lines[0].strip().lower() == _HEADER:
        lines = lines[1:]
    if not lines or len(lines) > 20:
        raise ValueError("ReDial output requires one to twenty ranked items")
    titles: list[str] = []
    for expected_rank, line in enumerate(lines, start=1):
        match = _RANKED_LINE.fullmatch(line)
        rank_match = _RANK.match(line)
        if match is None or rank_match is None or int(rank_match.group("rank")) != expected_rank:
            raise ValueError("ReDial ranks must be sequential and every line must be an item")
        title = match.group("title").strip(" \t")
        if not title:
            raise ValueError("ReDial movie title must not be empty")
        titles.append(title)
    return tuple(titles)


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def clean_prediction_title(value: str) -> str:
    return _collapse_whitespace(_PARENTHETICAL.sub("", value))


def clean_catalog_title(uri: str) -> str:
    path = urlsplit(uri).path or uri
    segment = unquote(path.rsplit("/", 1)[-1])
    separated = segment.replace("_", " ").replace("-", " ").replace(">", " ")
    return _collapse_whitespace(_PARENTHETICAL.sub("", separated))


def normalize_movie_title(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def resolve_ranked_movies(
    raw_titles: tuple[str, ...],
    catalog: MovieCatalog,
) -> tuple[ResolvedMovie, ...]:
    resolved: list[ResolvedMovie] = []
    ordered_catalog = sorted(
        catalog.movies,
        key=lambda movie: (movie.normalized_title.encode("utf-8"), movie.entity_id),
    )
    for raw_title in raw_titles:
        cleaned = clean_prediction_title(raw_title)
        normalized = normalize_movie_title(cleaned)
        if not normalized:
            raise ValueError("ReDial prediction has an empty cleaned title")
        distances = tuple(
            (_levenshtein(normalized, movie.normalized_title), movie) for movie in ordered_catalog
        )
        minimum = min(distance for distance, _movie in distances)
        tied = tuple(movie for distance, movie in distances if distance == minimum)
        chosen = tied[0]
        resolved.append(
            ResolvedMovie(
                raw_title=raw_title,
                cleaned_title=cleaned,
                normalized_title=normalized,
                entity_id=chosen.entity_id,
                catalog_title=chosen.cleaned_title,
                catalog_normalized_title=chosen.normalized_title,
                distance=minimum,
                tied_entity_ids=tuple(movie.entity_id for movie in tied),
            )
        )
    return tuple(resolved)


def score_recall_at_5(
    resolved: tuple[ResolvedMovie, ...],
    gold_entity_ids: tuple[int, ...],
    catalog: MovieCatalog,
) -> RecallAtFiveResult:
    if not gold_entity_ids:
        raise ValueError("ReDial Recall@5 requires at least one gold entity")
    by_id = {movie.entity_id: movie for movie in catalog.movies}
    try:
        gold_titles = tuple(by_id[entity_id].normalized_title for entity_id in gold_entity_ids)
    except KeyError as exc:
        raise ValueError(f"gold ReDial entity is absent from catalog: {exc.args[0]}") from exc
    predicted = tuple(movie.catalog_normalized_title for movie in resolved[:5])
    numerator = sum(title in predicted for title in gold_titles)
    denominator = len(gold_titles)
    return RecallAtFiveResult(
        numerator=numerator,
        denominator=denominator,
        value=Fraction(numerator, denominator),
        gold_entity_ids=gold_entity_ids,
        gold_normalized_titles=gold_titles,
        predicted_normalized_titles=predicted,
    )
