# Open Agent Memory v0.0.1: BEAM-10M

## Overview

Open the [interactive offline report](report.html) to inspect all 200 questions with OAM and Hindsight saved-context replay side by side, including complete injected contexts, answers, rubrics, judgments and retries. Its AMB-inspired layout supports category, text and outcome filters and separate first-pass / policy-final views. Download `report.html` and open it directly in a current browser; no setup or evaluation calls are needed. Full request/response archive links use the adjacent evidence files. See [rebuild instructions](../README.md).

The saved OAM BEAM-10M evaluation covers **10 complete conversations and 200 distinct questions**. Its policy-final mean rubric score is **61.56%**, and **142/200 questions (71.0%)** score at least 0.5. The first-pass mean is **59.89%**, with **136/200** threshold passes. Mean rubric score and threshold pass rate measure different things.

This report reuses the completed November 2026 development campaign. It makes no new ingestion, retrieval, answer, judge or review calls. The result describes the saved OAM configurations and contexts, not a fresh test of the final release artifact.

## 1. Evaluation and results

BEAM-10M tests memory over long conversation histories using questions with rubric items. Each question's score is the mean of its rubric-item scores; the reported mean weights questions equally. A threshold pass means a question score of at least 0.5. Original questions, gold and rubrics remain in the denominator, including disputed cases.

| Result | First pass | Policy final |
| --- | ---: | ---: |
| OAM mean rubric score | 0.598923250 | 0.615562139 |
| OAM threshold passes | 136/200 | 142/200 |
| Hindsight saved-context replay mean | 0.504018308 | 0.521573864 |
| Hindsight saved-context replay threshold passes | 108/200 | 112/200 |

First answers and judges used high reasoning. An eligible failure could receive one max-effort answer on the unchanged saved context and prompt, followed by high-effort judging; the dispatched retry determines the policy-final result even when its score falls. OAM used 44 such retries, and the saved-context replay used 43.

The comparison reuses historical Hindsight contexts and applies the same recorded answer and judge policy to the same question set. It does not rerun Hindsight indexing or retrieval and is not a fresh matched engine comparison. It does not isolate the effect of either memory engine, nor establish a comparison to results produced with different models or scoring policies.

## 2. Configuration and evidence

| Setting | Recorded campaign |
| --- | --- |
| Dataset | BEAM-10M, 10 conversations and 200 unique questions. |
| Dataset revision | `9b2096193fe74e2837e4713e483351e19817773c` |
| Dataset SHA-256 | `c16643a7cc7d93c1ed5509568cdd9e4c387dcd60414e68dde740b54064e68ba8` |
| OAM path | Native L1 indexing and normal recall; extraction reasoning low; Profile disabled; L2 excluded from scored contexts. |
| Embedding and tokenization | Qwen3 embedding with a 32K-token input limit; OAM used its approximate tokenizer during the evaluation. |
| Answer and judge | DeepSeek Flash; high-effort first answer and judge, with one eligible max-effort answer retry. |
| Date convention | Source batch reference dates; one previously approved next-batch date imputation for the first undated batch of conversation 7. |
| Evaluation evidence | Saved question selections, contexts, answers, judgments, retries, source-assisted failure reviews and validation receipts. |

All 200 OAM first-pass answers and judgments and approximate full-context noise assessments were retained. All 58 policy-final failures received source-assisted source/index/context/answer/judge reviews under the available-evidence method. Of those reviews, 57 have canonical exact noise ratios and one has an explicitly uncertain ratio. Historical stages that were unavailable, ambiguous gold and incomplete billing remain disclosed limitations; they are not assumed to be successful or zero.

The saved final summary SHA-256 is `40564e93d22a316ba77d7763bb25a345d54af1ceafa1853be6d9ca982779e9f5`; the paired policy snapshot SHA-256 is `1c2da27f3c9677223ea0b41641203c90ce70b776b568a6e89561bb2a2db60c99`. A content-bearing public projection is now available below. The original private transport and operational artifacts remain separate; their hashes identify checked source artifacts rather than certifying a new evaluation.

## 3. Performance and release boundary

The recorded OAM answer-visible contexts are 17.43% larger than the paired historical Hindsight contexts, exceeding the campaign's 10% overhead target. The comparison does not establish a matched indexing-token, indexing-time or retrieval-latency pass. The complete campaign's cost is unknown, and incomplete or failed usage cannot be treated as zero. Embedding tokens are excluded from generation-token comparisons.

For conversations 2 through 10, 540 saved public-HTTP recall samples across 180 questions have a mean of 2,906.5 ms, a median of 2,617.9 ms and a p95 of 5,527.6 ms; no samples were excluded. The first conversation's historical timings are separate because its later retained index state differs. These timings are descriptive, not a matched speed comparison.

Evaluation completion does not certify OAM v0.0.1 product acceptance. In particular, the saved L1-only evaluation does not prove L2, Profile, document Resource, multi-replica or release-artifact behavior. The companion [LME-500 report](../lme500/README.md) describes a separate fast-recall campaign and its own snapshot boundaries.

## 4. Public per-question evidence

The [manifest](manifest.json) lists ten compressed conversation shards for OAM and ten for the Hindsight saved-context replay. Each arm covers the same 200 distinct question IDs. Every record includes the original question, gold and rubric, exact saved context and its SHA-256, `cl100k_base` context-token count, recorded retrieval time, first-pass answer and score, any selected max-effort retry, and policy-final answer and score.

`first_pass` and `retry` preserve both outcomes; `policy_final_attempt` follows the frozen paired-policy snapshot and selects the dispatched retry even when its score falls. The OAM and replay arms retain 44 and 43 retries, respectively. `correct` means rubric score at least 0.5; the mean rubric score is computed separately. Retrieval times remain their saved measurements and do not establish a matched engine comparison.

All 400 first-pass attempts and 87 selected retries include their actual saved `answer_call` request, visible response, usage and source receipt hash. The 1,495 ordered `judgments` retain each rubric criterion, raw and normalized score, reason, actual judge request and visible response. Each receipt is matched to the saved completed-input identity and output; duplicate transport attempts are not counted as extra rubric items. Applying the original 0/0.5/1 normalization and averaging the rubric scores reproduces every saved per-question score. The frozen policy snapshot independently confirms which attempt contributes to the reported result. The manifest also records saved runtime revisions, configuration/evaluator hashes and sanitized answer/judge bindings where available; unavailable revision metadata remains null.

The data retain [BEAM attribution and CC BY-SA 4.0 terms](LICENSES/BEAM-CC-BY-SA-4.0.md), including the applicable separate prompt notices. The two metadata-adjudicated retries retain their original schema-invalid visible responses and recorded answer projections, with unchanged answer text and no replacement answer call. Full conversation shards, private service credentials, transport headers, hidden `reasoning_content` and operational logs are not included; visible reasoning fields explicitly requested by the benchmark remain in the response evidence. These files support question-level inspection and independent rescoring; they do not recreate historical indexing or certify a fresh matched run.

From the repository root, run `uv run pytest -q tests/test_published_oam_evidence.py` to recompute both arms' question coverage, first-pass and policy-final scores, threshold counts, retry selection and exact context-token counts. Python's standard `gzip` and `json` modules open the shards without provider credentials.
