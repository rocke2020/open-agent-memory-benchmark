"""Published OAM model requests and structured answers."""

from __future__ import annotations

import json
from typing import Any

from oamb.workloads.model_parameters import model_request_parameters

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Step-by-step explanation of how you used the context to arrive at the answer.",
        },
        "answer": {
            "type": "string",
            "description": "The final concise answer to the question. If the context lacks the information, say so.",
        },
    },
    "required": ["reasoning", "answer"],
    "additionalProperties": False,
}
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "description": "Score: 0.0, 0.5, or 1.0"},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}


def completion_payload(
    model: str, effort: str, prompt: str, schema: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        **model_request_parameters(model, effort, thinking="enabled" if schema else None),
        "messages": [{"role": "user", "content": prompt}],
    }
    if schema is None:
        payload.update(n=1, temperature=0, top_p=1, stop=None)
    else:
        payload["messages"].insert(
            0,
            {
                "role": "system",
                "content": "Return only a JSON object matching this schema: " + json.dumps(schema),
            },
        )
        payload["response_format"] = {"type": "json_object"}
    return payload


def answer_text(content: str, dataset: str) -> str:
    if dataset == "lme500":
        if not content.strip():
            raise ValueError("empty answer")
        return content.strip()
    try:
        value = json.loads(content)
    except ValueError:
        raise ValueError("BEAM answer must contain reasoning and answer JSON") from None
    if (
        not isinstance(value, dict)
        or set(value) != {"reasoning", "answer"}
        or not isinstance(value["reasoning"], str)
        or not isinstance(value["answer"], str)
        or not value["answer"].strip()
    ):
        raise ValueError("BEAM answer must contain reasoning and a nonempty answer")
    return str(value["answer"]).strip()
