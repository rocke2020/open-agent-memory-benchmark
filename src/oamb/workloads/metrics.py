"""Pure LongMemEval output parsers and metric contracts."""

from __future__ import annotations

from types import MappingProxyType

from oamb.contracts.ports import FinishDisposition, ModelCompletion
from oamb.contracts.specifications import MetricSpec, OutputContract

_ASCII_TRIM = " \t\r\n"

LME_ANSWER_MAX_OUTPUT_TOKENS = 8_192
LME_JUDGE_MAX_OUTPUT_TOKENS = 1024

OUTPUT_CONTRACTS = MappingProxyType(
    {
        contract.output_contract_id: contract
        for contract in (
            OutputContract(
                output_contract_id="lme-answer-text-v1",
                representation="text",
                parser_id="lme-answer-text-v1",
                max_output_tokens=LME_ANSWER_MAX_OUTPUT_TOKENS,
                required_candidate_count=1,
                accepted_finish_dispositions=("normal_stop",),
                parse_failure_policy="terminal_error",
            ),
            OutputContract(
                output_contract_id="lme-judge-yes-no-v1",
                representation="boolean",
                parser_id="lme-judge-yes-no-v1",
                max_output_tokens=LME_JUDGE_MAX_OUTPUT_TOKENS,
                required_candidate_count=1,
                accepted_finish_dispositions=("normal_stop",),
                parse_failure_policy="unjudged",
            ),
        )
    }
)

METRIC_SPECS = MappingProxyType(
    {
        metric.metric_id: metric
        for metric in (
            MetricSpec(
                metric_id="lme-judged-accuracy-v1",
                output_contract_id="lme-judge-yes-no-v1",
                normalizer_id="ascii-trim-lower-v1",
                scorer_id="lme-judge-yes-no-v1",
                answer_set_policy="single_rubric",
                input_fields=("question", "gold_or_rubric", "visible_answer"),
                judge_prompt_pack_id="oamb-lme-judge-v1",
                failure_semantics="unavailable",
                unjudged_semantics="unjudged",
            ),
        )
    }
)


def parse_lme_answer_text(raw_output: str) -> str:
    if not isinstance(raw_output, str) or not raw_output.strip(_ASCII_TRIM):
        raise ValueError("lme answer must be non-empty text")
    return raw_output


def parse_lme_judge_yes_no(raw_output: str) -> bool:
    if not isinstance(raw_output, str):
        raise ValueError("lme judge output must be text")
    value = raw_output.strip(_ASCII_TRIM)
    try:
        lowered = value.encode("ascii").decode("ascii").lower()
    except UnicodeEncodeError as exc:
        raise ValueError("lme judge output must be ASCII yes or no") from exc
    if lowered == "yes":
        return True
    if lowered == "no":
        return False
    raise ValueError("lme judge output must be exactly yes or no")


def _single_normal_text(completion: ModelCompletion) -> str:
    if completion.finish_disposition != FinishDisposition.NORMAL_STOP:
        raise ValueError("output requires a normal stop")
    if len(completion.candidates) != 1:
        raise ValueError("output requires exactly one candidate")
    candidate = completion.candidates[0]
    if candidate.content is None or candidate.tool_call_present or not candidate.complete:
        raise ValueError("output requires complete text without a tool call")
    return candidate.content


def parse_lme_answer_completion(completion: ModelCompletion) -> str:
    return parse_lme_answer_text(_single_normal_text(completion))


def parse_lme_judge_completion(completion: ModelCompletion) -> bool:
    return parse_lme_judge_yes_no(_single_normal_text(completion))
