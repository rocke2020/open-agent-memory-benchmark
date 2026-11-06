# Open Agent Memory v0.0.1: BEAM-10M

## Overview

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

The saved final summary SHA-256 is `40564e93d22a316ba77d7763bb25a345d54af1ceafa1853be6d9ca982779e9f5`; the paired policy snapshot SHA-256 is `1c2da27f3c9677223ea0b41641203c90ce70b776b568a6e89561bb2a2db60c99`. The detailed source artifacts are retained privately; these hashes identify the checked local artifacts and do not themselves make the underlying records publicly reproducible.

## 3. Performance and release boundary

The recorded OAM answer-visible contexts are 17.43% larger than the paired historical Hindsight contexts, exceeding the campaign's 10% overhead target. The comparison does not establish a matched indexing-token, indexing-time or retrieval-latency pass. The complete campaign's cost is unknown, and incomplete or failed usage cannot be treated as zero. Embedding tokens are excluded from generation-token comparisons.

For conversations 2 through 10, 540 saved public-HTTP recall samples across 180 questions have a mean of 2,906.5 ms, a median of 2,617.9 ms and a p95 of 5,527.6 ms; no samples were excluded. The first conversation's historical timings are separate because its later retained index state differs. These timings are descriptive, not a matched speed comparison.

Evaluation completion does not certify OAM v0.0.1 product acceptance. In particular, the saved L1-only evaluation does not prove L2, Profile, document Resource, multi-replica or release-artifact behavior. The companion [LME-500 report](oam-v0.0.1-lme500.md) describes a separate fast-recall campaign and its own snapshot boundaries.
