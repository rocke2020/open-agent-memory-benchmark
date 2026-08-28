from __future__ import annotations

import hashlib
import tracemalloc
from pathlib import Path

import pytest

from oamb.reporting.previews import build_display_preview


def test_display_preview_streams_full_identity_but_keeps_only_utf8_bounded_text(
    tmp_path: Path,
) -> None:
    payload = ("évidence-" * 32).encode()
    source = tmp_path / "large-evidence.json"
    source.write_bytes(payload)

    preview = build_display_preview(
        source,
        source_reference="source/raw/fixture.json",
        media_type="application/json",
        max_bytes=17,
    )

    assert preview.total_bytes == len(payload)
    assert preview.shown_bytes <= 17
    assert preview.text.encode() == payload[: preview.shown_bytes]
    assert preview.sha256 == hashlib.sha256(payload).hexdigest()
    assert preview.truncated is True
    assert preview.limitation == "preview truncated at 17 UTF-8 bytes"


def test_display_preview_rejects_invalid_utf8_and_nonpositive_limit(tmp_path: Path) -> None:
    source = tmp_path / "invalid.bin"
    source.write_bytes(b"valid-prefix\xffinvalid")

    with pytest.raises(ValueError, match="UTF-8"):
        build_display_preview(
            source,
            source_reference="source/raw/invalid.bin",
            media_type="application/octet-stream",
            max_bytes=8,
        )
    with pytest.raises(ValueError, match="positive"):
        build_display_preview(
            source,
            source_reference="source/raw/invalid.bin",
            media_type="application/octet-stream",
            max_bytes=0,
        )


def test_display_preview_large_source_has_bounded_python_allocation(tmp_path: Path) -> None:
    source = tmp_path / "large.json"
    source.write_bytes(b"x" * (8 * 1024 * 1024))

    tracemalloc.start()
    try:
        preview = build_display_preview(
            source,
            source_reference="source/raw/large.json",
            media_type="application/json",
            max_bytes=4_096,
        )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert preview.total_bytes == 8 * 1024 * 1024
    assert preview.shown_bytes == 4_096
    assert peak < 512 * 1024
