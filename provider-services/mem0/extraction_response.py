"""Normalize text-bearing extraction entries in the pinned Mem0 source build."""

import json
import sys
from pathlib import Path
from typing import Any

_EMPTY_CHECK = "        if not extracted_memories:\n"
_IMPLEMENTATION_COUNT = 2
_IMPORT = (
    "from mem0.memory.oamb_extraction_response import "
    "generate_parse_normalize_with_retry, normalize_extracted_memories\n"
)
_NORMALIZE = "        extracted_memories = normalize_extracted_memories(extracted_memories)\n"
_SYNC_START = "        try:\n            response = self.llm.generate_response(\n"
_ASYNC_START = "        try:\n            response = await asyncio.to_thread(\n"
_PARSE_END = "\n        if not extracted_memories:\n"
_SYNC_REPLACEMENT = """        extracted_memories = generate_parse_normalize_with_retry(
            lambda: self.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            ),
            remove_code_blocks=remove_code_blocks,
            extract_json=extract_json,
            max_retries=int(os.environ["OAMB_EXTRACTION_MAX_RETRIES"]),
        )
"""
_ASYNC_REPLACEMENT = """        extracted_memories = await asyncio.to_thread(
            generate_parse_normalize_with_retry,
            lambda: self.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            ),
            remove_code_blocks=remove_code_blocks,
            extract_json=extract_json,
            max_retries=int(os.environ["OAMB_EXTRACTION_MAX_RETRIES"]),
        )
"""
_ERROR_CLASSIFICATION_FALLBACK = '    return ("unknown", "Upstream provider error.")\n'
_EXTRACTION_ERROR_CLASSIFICATION = """    if name == "ExtractionResponseError":
        return (
            "provider_extraction_failed",
            "Provider memory extraction failed after retries.",
        )
    return ("unknown", "Upstream provider error.")
"""


class ExtractionResponseError(RuntimeError):
    """The native extraction call never produced a valid memory payload."""


def generate_parse_normalize_with_retry(
    generate_response: Any,
    *,
    remove_code_blocks: Any,
    extract_json: Any,
    max_retries: int,
) -> list[dict[str, Any]]:
    """Retry one Mem0 extraction generation through validated parsing."""
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError("Mem0 extraction max_retries must be a non-negative integer")
    last_error: Exception | None = None
    for _attempt in range(max_retries + 1):
        try:
            response = remove_code_blocks(generate_response())
            if not isinstance(response, str) or not response.strip():
                raise ValueError("Mem0 extraction response must be nonblank JSON")
            try:
                document = json.loads(response, strict=False)
            except json.JSONDecodeError:
                document = json.loads(extract_json(response), strict=False)
            if not isinstance(document, dict) or "memory" not in document:
                raise ValueError("Mem0 extraction response requires a memory field")
            return normalize_extracted_memories(document["memory"])
        except Exception as exc:
            last_error = exc
    raise ExtractionResponseError(
        f"Mem0 extraction failed after {max_retries + 1} attempts"
    ) from last_error


def normalize_extracted_memories(value: object) -> list[dict[str, Any]]:
    """Preserve complete native entries and wrap extracted text without inventing fields."""
    if isinstance(value, (str, dict)):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError("Mem0 extraction memory must contain text entries")

    normalized: list[dict[str, Any]] = []
    for entry in values:
        if isinstance(entry, str):
            item = {"text": entry}
        elif isinstance(entry, dict):
            item = entry
        else:
            raise ValueError("Mem0 extraction memory entry must be text or an object")
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Mem0 extraction memory entry requires nonblank text")
        normalized.append(item)
    return normalized


def patch_source(source: str) -> str:
    """Patch both pinned extraction stages, rejecting unexpected source drift."""
    if _IMPORT in source or source.count(_EMPTY_CHECK) != _IMPLEMENTATION_COUNT:
        raise ValueError("Mem0 extraction parser source differs from the pinned patch target")
    if source.count(_SYNC_START) == 1 and source.count(_ASYNC_START) == 1:
        for start, replacement in (
            (_SYNC_START, _SYNC_REPLACEMENT),
            (_ASYNC_START, _ASYNC_REPLACEMENT),
        ):
            prefix, remainder = source.split(start, 1)
            _discarded, suffix = remainder.split(_PARSE_END, 1)
            source = prefix + replacement + _PARSE_END + suffix
        return _IMPORT + source
    if _SYNC_START not in source and _ASYNC_START not in source:
        return _IMPORT + source.replace(_EMPTY_CHECK, _NORMALIZE + _EMPTY_CHECK)
    raise ValueError("Mem0 extraction generation source differs from the pinned patch target")


def patch_errors_source(source: str) -> str:
    """Give exhausted extraction its stable pinned REST error classification."""
    if source.count(_ERROR_CLASSIFICATION_FALLBACK) != 2:
        raise ValueError("Mem0 server error source differs from the pinned patch target")
    return source.replace(
        _ERROR_CLASSIFICATION_FALLBACK,
        _EXTRACTION_ERROR_CLASSIFICATION,
        1,
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: extraction_response.py MEM0_MAIN_PY MEM0_ERRORS_PY")
    target = Path(sys.argv[1])
    patched = patch_source(target.read_text(encoding="utf-8"))
    target.write_text(patched, encoding="utf-8")
    errors_target = Path(sys.argv[2])
    patched_errors = patch_errors_source(errors_target.read_text(encoding="utf-8"))
    errors_target.write_text(patched_errors, encoding="utf-8")
