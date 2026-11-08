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

The archived upstream Hindsight result used **Gemini 3.1 Pro for answers and Gemini 2.5 Flash Lite for judging**, reporting **64.08% mean rubric score and 147/200 threshold passes**. Our **52.16% DeepSeek saved-context replay** is a different answer/judge experiment. Comparing OAM's 61.56% against this replay does not establish that OAM exceeds the upstream result; preserve the model and scoring-policy distinctions.

### 1.1 Results by conversation

Each conversation contributes the same 20 questions in both arms. Mean rubric scores are percentages; pass counts use the 0.5 threshold. These rows use the complete published policy snapshot, including every selected retry.

| Conversation | OAM first mean | Replay first mean | OAM policy-final mean | Replay policy-final mean | OAM passes: first → final | Replay passes: first → final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 60.1288% | 37.2917% | 63.8788% | 44.5417% | 14/20 → 15/20 | 9/20 → 10/20 |
| 2 | 45.2917% | 49.2917% | 54.2500% | 49.2917% | 10/20 → 14/20 | 11/20 → 11/20 |
| 3 | 54.8750% | 52.7500% | 54.6250% | 53.7500% | 12/20 → 12/20 | 10/20 → 10/20 |
| 4 | 73.0871% | 59.1919% | 73.0871% | 59.8169% | 16/20 → 16/20 | 13/20 → 14/20 |
| 5 | 52.0000% | 40.7500% | 52.0000% | 40.7500% | 13/20 → 13/20 | 9/20 → 9/20 |
| 6 | 72.1865% | 60.5556% | 71.4921% | 61.1111% | 15/20 → 15/20 | 14/20 → 14/20 |
| 7 | 60.4375% | 48.1250% | 60.9375% | 53.1250% | 14/20 → 14/20 | 12/20 → 13/20 |
| 8 | 59.5417% | 45.5625% | 58.7917% | 47.2292% | 13/20 → 13/20 | 8/20 → 8/20 |
| 9 | 51.7500% | 43.1250% | 52.7500% | 43.9583% | 13/20 → 13/20 | 9/20 → 9/20 |
| 10 | 69.6250% | 67.3750% | 73.7500% | 68.0000% | 16/20 → 17/20 | 13/20 → 14/20 |

A high-to-max recovery is an observed outcome, not proof that reasoning effort caused the improvement. The policy also retains score decreases; it never chooses the better attempt.

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

The [complete failure review](beam10m-failure-review.md) assigns one primary cause per failed question: 30 ambiguous gold answers, 2 incorrect gold answers, 14 retrieval misses, 1 answer-handling error and 11 unresolved. Reference-answer issues account for **32/58 failures (55.2%)** under this review; overlapping secondary findings remain separate. Official scores and the 200-question denominator are unchanged. The public copy retains the detailed analysis and embedded evidence; unpublished source/index audit artifacts are identified by filename rather than inaccessible local links.

The saved final summary SHA-256 is `40564e93d22a316ba77d7763bb25a345d54af1ceafa1853be6d9ca982779e9f5`; the paired policy snapshot SHA-256 is `1c2da27f3c9677223ea0b41641203c90ce70b776b568a6e89561bb2a2db60c99`. A content-bearing public projection is now available below. The original private transport and operational artifacts remain separate; their hashes identify checked source artifacts rather than certifying a new evaluation.

## 3. Performance and release boundary

The recorded OAM answer-visible contexts are 17.43% larger than the paired historical Hindsight contexts, exceeding the campaign's 10% overhead target. The comparison does not establish a matched indexing-token, indexing-time or retrieval-latency pass. The complete campaign's cost is unknown, and incomplete or failed usage cannot be treated as zero. Embedding tokens are excluded from generation-token comparisons.

For conversations 2 through 10, 540 saved public-HTTP recall samples across 180 questions have a mean of 2,906.5 ms, a median of 2,617.9 ms and a p95 of 5,527.6 ms; no samples were excluded. The first conversation's historical timings are separate because its later retained index state differs. These timings are descriptive, not a matched speed comparison.

Evaluation completion does not certify OAM v0.0.1 product acceptance. In particular, the saved L1-only evaluation does not prove L2, Profile, document Resource, multi-replica or release-artifact behavior. The companion [LME-500 report](../lme500/README.md) describes a separate fast-recall campaign and its own snapshot boundaries.

### 3.1 Saved recall and context by conversation

Each row uses the 20 saved question records in each arm and one retained successful recall per question. Timings come from different runtime/host/model-serving conditions and are descriptive; they do not establish a controlled speed advantage. Earlier failed attempts and the separate 540-probe series are not substituted into these means. Context tokens count the complete answer-visible text, including formatting and provenance, with `cl100k_base`.

| Conversation | OAM mean saved recall | HS historical mean recall | OAM mean context tokens | Replay mean context tokens | OAM context overhead |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.341 s | 7.999 s | 31,164.60 | 27,571.05 | +13.03% |
| 2 | 2.918 s | 9.700 s | 30,279.35 | 26,968.80 | +12.28% |
| 3 | 3.300 s | 8.375 s | 30,736.25 | 28,384.65 | +8.28% |
| 4 | 2.937 s | 11.426 s | 38,519.10 | 26,669.85 | +44.43% |
| 5 | 2.959 s | 9.979 s | 36,833.30 | 27,053.65 | +36.15% |
| 6 | 2.680 s | 9.986 s | 30,292.75 | 27,089.75 | +11.82% |
| 7 | 2.752 s | 16.741 s | 31,370.90 | 27,110.60 | +15.71% |
| 8 | 2.301 s | 15.905 s | 30,981.55 | 27,166.50 | +14.04% |
| 9 | 2.711 s | 8.721 s | 31,041.45 | 27,773.25 | +11.77% |
| 10 | 2.948 s | 48.618 s | 29,849.55 | 27,631.75 | +8.03% |
| **All 200 questions** | **2.785 s** | **14.745 s** | **32,106.88** | **27,341.99** | **+17.43%** |

The complete answer-visible contexts total **6,421,376 OAM tokens versus 5,468,397 replay tokens**. This is the answer input, not indexing usage or total provider cost.

### 3.2 Context quality and efficiency limits

Across the 57 failed-question contexts with exact source-assisted labels, unrelated body-text tokens total **431,958/1,826,803 (23.65%)**. The remaining failure has an uncertain ratio and is excluded. This is a failure-selected subset, not the full-200 noise rate; approximate model estimates are not mixed with exact span labels, and noise alone does not establish why an answer failed.

| Measure | Evidence / limit |
| --- | --- |
| Answer-visible context overhead | +17.43%; exceeds the campaign's +10% target. |
| Matched retrieval latency | Unverified; historical timings do not establish the +15% comparison target. |
| Matched generation-only indexing tokens | Unverified against the +20% target; embedding tokens are excluded. |
| Matched indexing elapsed time | Unverified against the +20% target; resumed readiness checks and historical ingestion metadata do not establish complete comparable durations. |
| Monetary cost / complete billing | Unknown; retained usage is not an invoice-quality total. |

### 3.3 Retained answer/judge generation usage

These totals sum unique completed answer/judge receipts published with all 200 questions in each arm. Generation tokens mean input plus output; reasoning is already included in output, and cached input is already included in input. They exclude native indexing, embeddings, source/context reviews, additional failed or unknown attempts not represented in these public receipts, and assistant/reviewer inference. They do not establish total campaign spend.

| Receipt scope | OAM calls / generation tokens | Replay calls / generation tokens | Included reasoning tokens: OAM / replay |
| --- | ---: | ---: | ---: |
| First-pass answers and judges | 821 / 7,732,359 | 821 / 6,525,773 | 506,377 / 416,118 |
| Selected max answers and high-effort judges | 180 / 1,870,126 | 160 / 1,546,459 | 263,173 / 216,877 |

The historical Hindsight archive records 5,504,578.7 ms of ingestion with `ingested_docs=1`; that field alone does not establish complete matched indexing duration. Source-recovery subtotals and calendar time across shutdowns/resumes are likewise not substitutes for full-cohort indexing usage or elapsed time.

## 4. Public per-question evidence

The [manifest](manifest.json) lists ten compressed conversation shards for OAM and ten for the Hindsight saved-context replay. Each arm covers the same 200 distinct question IDs. Every record includes the original question, gold and rubric, exact saved context and its SHA-256, `cl100k_base` context-token count, recorded retrieval time, first-pass answer and score, any selected max-effort retry, and policy-final answer and score.

`first_pass` and `retry` preserve both outcomes; `policy_final_attempt` follows the frozen paired-policy snapshot and selects the dispatched retry even when its score falls. The OAM and replay arms retain 44 and 43 retries, respectively. `correct` means rubric score at least 0.5; the mean rubric score is computed separately. Retrieval times remain their saved measurements and do not establish a matched engine comparison.

All 400 first-pass attempts and 87 selected retries include their actual saved `answer_call` request, visible response, usage and source receipt hash. The 1,495 ordered `judgments` retain each rubric criterion, raw and normalized score, reason, actual judge request and visible response. Each receipt is matched to the saved completed-input identity and output; duplicate transport attempts are not counted as extra rubric items. Applying the original 0/0.5/1 normalization and averaging the rubric scores reproduces every saved per-question score. The frozen policy snapshot independently confirms which attempt contributes to the reported result. The manifest also records saved runtime revisions, configuration/evaluator hashes and sanitized answer/judge bindings where available; unavailable revision metadata remains null.

The data retain [BEAM attribution and CC BY-SA 4.0 terms](LICENSES/BEAM-CC-BY-SA-4.0.md), including the applicable separate prompt notices. The two metadata-adjudicated retries retain their original schema-invalid visible responses and recorded answer projections, with unchanged answer text and no replacement answer call. Full conversation shards, private service credentials, transport headers, hidden `reasoning_content` and operational logs are not included; visible reasoning fields explicitly requested by the benchmark remain in the response evidence. These files support question-level inspection and independent rescoring; they do not recreate historical indexing or certify a fresh matched run.

From the repository root, run `uv run pytest -q tests/test_published_oam_evidence.py` to recompute both arms' question coverage, first-pass and policy-final scores, threshold counts, retry selection and exact context-token counts. Python's standard `gzip` and `json` modules open the shards without provider credentials.
