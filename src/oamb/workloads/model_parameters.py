"""Shared scoring-model reasoning adapter, ported from the OAM evaluator.

Copyright (c) 2026 Rocke Dong. MIT; see LICENSE.OAM-eval.
"""

from typing import Any, Literal

ReasoningEffort = Literal["none", "low", "high", "max", "adaptive"]
REASONING_EFFORTS = frozenset({"none", "low", "high", "max", "adaptive"})


def model_request_parameters(
    model_id: str, effort: str, *, thinking: str | None = None
) -> dict[str, Any]:
    if effort not in REASONING_EFFORTS:
        raise ValueError("unsupported benchmark reasoning effort")
    if effort in {"none", "adaptive"}:
        parameters: dict[str, Any] = {
            "thinking": {"type": "disabled" if effort == "none" else "adaptive"}
        }
    else:
        parameters = {"reasoning_effort": effort}
        if thinking is not None:
            parameters["thinking"] = {"type": thinking}
    if model_id.lower().rsplit("/", 1)[-1].startswith("minimax-"):
        parameters["reasoning_split"] = True
        if effort != "none":
            parameters["thinking"] = {"type": "adaptive"}
    return parameters
