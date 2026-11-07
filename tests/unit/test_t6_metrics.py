from __future__ import annotations

import pytest

from oamb.contracts.ports import FinishDisposition, ModelCandidate, ModelCompletion
from oamb.workloads.metrics import (
    parse_lme_answer_completion,
    parse_lme_answer_text,
    parse_lme_judge_completion,
    parse_lme_judge_yes_no,
)


def test_lme_output_parsers_are_strict_and_preserve_visible_answer_text() -> None:
    assert parse_lme_answer_text("  answer\n") == "  answer\n"
    assert parse_lme_judge_yes_no("\tYES\r\n") is True
    assert parse_lme_judge_yes_no(" no ") is False

    for invalid in ("", " \t\r\n", "yes.", "yesterday", "yes because"):
        with pytest.raises(ValueError):
            parse_lme_judge_yes_no(invalid)
    with pytest.raises(ValueError):
        parse_lme_answer_text(" \t\r\n")


def test_lme_completion_boundary_rejects_truncation_extra_candidates_and_tools() -> None:
    valid = ModelCompletion(
        finish_disposition=FinishDisposition.NORMAL_STOP,
        candidates=(ModelCandidate(content=" yes ", tool_call_present=False, complete=True),),
    )

    assert parse_lme_answer_completion(valid) == " yes "
    assert parse_lme_judge_completion(valid) is True
    invalid = (
        ModelCompletion(
            finish_disposition=FinishDisposition.LENGTH_LIMIT,
            candidates=valid.candidates,
        ),
        ModelCompletion(
            finish_disposition=FinishDisposition.NORMAL_STOP,
            candidates=valid.candidates * 2,
        ),
        ModelCompletion(
            finish_disposition=FinishDisposition.NORMAL_STOP,
            candidates=(ModelCandidate(content="yes", tool_call_present=True, complete=True),),
        ),
    )
    for completion in invalid:
        with pytest.raises(ValueError):
            parse_lme_answer_completion(completion)
