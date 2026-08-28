"""Bounded UTF-8 display previews over complete streamed source identities."""

from __future__ import annotations

import codecs
import gzip
import hashlib
from pathlib import Path
from typing import Literal

from oamb.contracts.reporting import DisplayPreview

PREVIEW_STREAM_CHUNK_BYTES = 64 * 1024


def build_display_preview(
    source_path: Path,
    *,
    source_reference: str,
    media_type: str,
    max_bytes: int,
    compression: Literal["none", "gzip"] = "none",
) -> DisplayPreview:
    """Hash and validate the full source while retaining at most ``max_bytes``."""

    if max_bytes < 1:
        raise ValueError("display preview byte limit must be positive")
    path = Path(source_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("display preview source must be a regular non-symlink file")

    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    retained = bytearray()
    total_bytes = 0
    try:
        with path.open("rb") as raw_source:
            source = (
                gzip.GzipFile(fileobj=raw_source, mode="rb")
                if compression == "gzip"
                else raw_source
            )
            while chunk := source.read(PREVIEW_STREAM_CHUNK_BYTES):
                digest.update(chunk)
                total_bytes += len(chunk)
                decoder.decode(chunk, final=False)
                if len(retained) < max_bytes:
                    retained.extend(chunk[: max_bytes - len(retained)])
            if source is not raw_source:
                source.close()
        decoder.decode(b"", final=True)
    except UnicodeDecodeError as exc:
        raise ValueError("display preview source is not valid UTF-8") from exc

    preview_bytes = bytes(retained)
    while preview_bytes:
        try:
            text = preview_bytes.decode("utf-8")
            break
        except UnicodeDecodeError as exc:
            if exc.reason != "unexpected end of data":
                raise ValueError("display preview source is not valid UTF-8") from exc
            preview_bytes = preview_bytes[: exc.start]
    else:
        text = ""

    shown_bytes = len(preview_bytes)
    truncated = shown_bytes < total_bytes
    return DisplayPreview(
        text=text,
        shown_bytes=shown_bytes,
        total_bytes=total_bytes,
        sha256=digest.hexdigest(),
        media_type=media_type,
        truncated=truncated,
        source_reference=source_reference,
        limitation=(f"preview truncated at {max_bytes} UTF-8 bytes" if truncated else None),
    )


__all__ = ["PREVIEW_STREAM_CHUNK_BYTES", "build_display_preview"]
