"""Pure workload output parsers and deterministic MemoryAgentBench metrics."""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
from types import MappingProxyType

from oamb.contracts.ports import FinishDisposition, ModelCompletion
from oamb.contracts.specifications import MetricSpec, OutputContract

_ASCII_TRIM = " \t\r\n"
_ANSWER_PREFIX = re.compile(r"answer:", flags=re.IGNORECASE | re.ASCII)
_ARTICLES = re.compile(r"\b(a|an|the)\b")

LME_ANSWER_MAX_OUTPUT_TOKENS = 8_192
LME_JUDGE_MAX_OUTPUT_TOKENS = 10
MAB_EVENTQA_MAX_OUTPUT_TOKENS = 40
MAB_ICL_MAX_OUTPUT_TOKENS = 20
MAB_REDIAL_MAX_OUTPUT_TOKENS = 512
MAB_DETECTIVEQA_MAX_OUTPUT_TOKENS = 2_000
MAB_FACTCONSOLIDATION_MAX_OUTPUT_TOKENS = 10

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
            OutputContract(
                output_contract_id="mab-first-line-v1",
                representation="text",
                parser_id="mab-first-line-v1",
                max_output_tokens=MAB_EVENTQA_MAX_OUTPUT_TOKENS,
                required_candidate_count=1,
                accepted_finish_dispositions=("normal_stop",),
                parse_failure_policy="terminal_error",
            ),
            OutputContract(
                output_contract_id="mab-raw-or-first-line-max-v1",
                representation="text",
                parser_id="mab-raw-or-first-line-max-v1",
                max_output_tokens=MAB_DETECTIVEQA_MAX_OUTPUT_TOKENS,
                required_candidate_count=1,
                accepted_finish_dispositions=("normal_stop",),
                parse_failure_policy="terminal_error",
            ),
            OutputContract(
                output_contract_id="mab-redial-ranked-movies-v1",
                representation="ranked_text_list",
                parser_id="mab-redial-ranked-movies-v1",
                max_output_tokens=MAB_REDIAL_MAX_OUTPUT_TOKENS,
                required_candidate_count=1,
                accepted_finish_dispositions=("normal_stop",),
                parse_failure_policy="terminal_error",
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
            MetricSpec(
                metric_id="mab-exact-v1",
                output_contract_id="mab-first-line-v1",
                compatible_output_contract_ids=("mab-raw-or-first-line-max-v1",),
                normalizer_id="mab-normalize-answer-v1",
                scorer_id="mab-exact-v1",
                answer_set_policy="max_over_preserved_alternatives",
                input_fields=("prediction", "gold_answers"),
                judge_prompt_pack_id=None,
                failure_semantics="unavailable",
                unjudged_semantics="not_applicable",
            ),
            MetricSpec(
                metric_id="mab-substring-em-v1",
                output_contract_id="mab-first-line-v1",
                compatible_output_contract_ids=("mab-raw-or-first-line-max-v1",),
                normalizer_id="mab-normalize-answer-v1",
                scorer_id="mab-substring-em-v1",
                answer_set_policy="max_over_preserved_alternatives",
                input_fields=("prediction", "gold_answers"),
                judge_prompt_pack_id=None,
                failure_semantics="unavailable",
                unjudged_semantics="not_applicable",
            ),
            MetricSpec(
                metric_id="mab-redial-recall-at-5-v1",
                output_contract_id="mab-redial-ranked-movies-v1",
                normalizer_id="mab-redial-title-normalizer-v1",
                scorer_id="mab-redial-recall-at-5-v1",
                answer_set_policy="preserve_occurrences",
                input_fields=("resolved_predictions", "gold_entity_ids"),
                judge_prompt_pack_id=None,
                failure_semantics="unavailable",
                unjudged_semantics="not_applicable",
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


def parse_mab_first_line(raw_output: str) -> str:
    if not isinstance(raw_output, str):
        raise ValueError("MAB output must be text")
    match = re.search(
        r"answer:(.*?)(?:\r\n|\r|\n|$)",
        raw_output,
        flags=re.IGNORECASE | re.ASCII,
    )
    if match is None:
        first_line = re.split(r"\r\n|\r|\n", raw_output, maxsplit=1)[0]
    else:
        first_line = match.group(1)
    parsed = first_line.strip()
    return _ANSWER_PREFIX.sub("", parsed, count=1).strip()


def parse_mab_raw_or_first_line_max(
    raw_output: str, gold_answers: Sequence[str]
) -> tuple[str, str]:
    if not gold_answers:
        raise ValueError("MAB metric requires at least one gold answer")
    return raw_output, parse_mab_first_line(raw_output)


def normalize_mab_answer(value: str) -> str:
    lowered = value.lower()
    without_punctuation = "".join(char for char in lowered if char not in string.punctuation)
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def _normalized_mab_ground_truth(value: str) -> str:
    normalized = normalize_mab_answer(value)
    if not normalized:
        raise ValueError("MAB ground truth must remain non-empty after normalization")
    return normalized


def mab_exact(prediction: str, ground_truth: str) -> bool:
    return normalize_mab_answer(prediction) == _normalized_mab_ground_truth(ground_truth)


def mab_substring_em(prediction: str, ground_truth: str) -> bool:
    return _normalized_mab_ground_truth(ground_truth) in normalize_mab_answer(prediction)


def max_over_gold_answers(
    prediction: str,
    gold_answers: Sequence[str],
    *,
    substring: bool,
) -> bool:
    if not gold_answers:
        raise ValueError("MAB metric requires at least one gold answer")
    normalized_prediction = normalize_mab_answer(prediction)
    normalized_gold_answers = tuple(_normalized_mab_ground_truth(gold) for gold in gold_answers)
    if substring:
        return any(gold in normalized_prediction for gold in normalized_gold_answers)
    return any(gold == normalized_prediction for gold in normalized_gold_answers)


def score_raw_or_first_line_max(
    raw_output: str,
    gold_answers: Sequence[str],
    *,
    substring: bool,
) -> bool:
    return any(
        max_over_gold_answers(candidate, gold_answers, substring=substring)
        for candidate in parse_mab_raw_or_first_line_max(raw_output, gold_answers)
    )
