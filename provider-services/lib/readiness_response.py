"""Strict minimum response validation for live-readiness probes."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


class ReadinessResponseError(ValueError):
    """A readiness endpoint returned an unusable or mismatched response."""


def validate_chat_response(document: object, expected_model: str) -> None:
    if not isinstance(document, dict) or document.get("model") != expected_model:
        raise ReadinessResponseError("chat response model does not match the requested model")
    choices = document.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ReadinessResponseError("chat response has no usable choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ReadinessResponseError("chat response has no assistant message")
    content = message.get("content")
    if message.get("role") != "assistant" or not isinstance(content, str) or not content.strip():
        raise ReadinessResponseError("chat response assistant content is empty or malformed")


def validate_embedding_response(
    document: object,
    expected_model: str,
    expected_dimension: int,
) -> None:
    if not isinstance(document, dict) or document.get("model") != expected_model:
        raise ReadinessResponseError("embedding response model does not match the requested model")
    data = document.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise ReadinessResponseError("embedding response has no vector")
    vector = data[0].get("embedding")
    if not isinstance(vector, list) or len(vector) != expected_dimension:
        raise ReadinessResponseError("embedding response dimension is invalid")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in vector
    ):
        raise ReadinessResponseError("embedding response contains a non-finite number")


def _load_document(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReadinessResponseError("readiness response is not valid JSON") from exc


def main() -> None:
    if len(sys.argv) not in {4, 5}:
        raise SystemExit(
            "usage: readiness_response.py chat RESPONSE EXPECTED_MODEL | "
            "embedding RESPONSE EXPECTED_MODEL DIMENSION"
        )
    kind, response_path, expected_model = sys.argv[1:4]
    document = _load_document(Path(response_path))
    try:
        if kind == "chat" and len(sys.argv) == 4:
            validate_chat_response(document, expected_model)
        elif kind == "embedding" and len(sys.argv) == 5:
            validate_embedding_response(document, expected_model, int(sys.argv[4]))
        else:
            raise ReadinessResponseError("unknown readiness response kind")
    except (ReadinessResponseError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
