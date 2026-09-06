"""Normalize text-bearing extraction entries in the pinned Mem0 source build."""

import sys
from pathlib import Path
from typing import Any

_EMPTY_CHECK = "        if not extracted_memories:\n"
_IMPLEMENTATION_COUNT = 2
_IMPORT = "from mem0.memory.oamb_extraction_response import normalize_extracted_memories\n"
_NORMALIZE = "        extracted_memories = normalize_extracted_memories(extracted_memories)\n"


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
    """Patch both pinned parsers after their broad exception handlers, with no fuzzy match."""
    if _IMPORT in source or source.count(_EMPTY_CHECK) != _IMPLEMENTATION_COUNT:
        raise ValueError("Mem0 extraction parser source differs from the pinned patch target")
    return _IMPORT + source.replace(_EMPTY_CHECK, _NORMALIZE + _EMPTY_CHECK)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: extraction_response.py MEM0_MAIN_PY")
    target = Path(sys.argv[1])
    patched = patch_source(target.read_text(encoding="utf-8"))
    target.write_text(patched, encoding="utf-8")
