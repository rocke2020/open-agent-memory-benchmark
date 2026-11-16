# Open Agent Memory: BEAM-10M

## Overview

The recorded selected OAM answers yield **61.56% mean rubric score**, with **142/200 questions (71.0%)** scoring at least 0.5. The result covers ten complete conversations and 200 distinct questions. Recorded selected-answer settings include high and max; these measurements are separate from the current DeepSeek max profile and are not a fresh max-profile evaluation. Exact answer settings are preserved in the evidence.

Open the [offline report](report.html) to inspect selected saved answers, complete contexts, references, rubrics and judgments for OAM and Hindsight saved-context replay side by side. Category, text and outcome filters work without a server or network. Full saved requests and responses are available in the adjacent compressed files. Packaging the report makes no model, retrieval, ingestion or database calls.

## 1. Recorded results

A question score is the mean of its rubric-item scores, and the reported mean weights all 200 questions equally. A threshold pass means a question score of at least 0.5. Mean score and threshold pass rate measure different things. Original questions, gold answers and rubrics remain in the denominator, including disputed cases.

| Arm | Mean rubric score | Questions scoring at least 0.5 |
| --- | ---: | ---: |
| OAM selected saved answers | **61.56%** | **142/200 (71.0%)** |
| Hindsight saved-context replay | 52.16% | 112/200 (56.0%) |

The replay uses historical Hindsight contexts with recorded DeepSeek answers and judgments. It does not rerun Hindsight indexing or retrieval and does not isolate the effect of either memory engine. The upstream Hindsight result used Gemini 3.1 Pro answers and Gemini 2.5 Flash Lite judging, reporting 64.08% mean score and 147/200 threshold passes. Those model settings differ from this replay, so OAM's result does not establish superiority over the upstream result.

### 1.1 Selected results by conversation

Both arms contain the same 20 questions per conversation.

| Conversation | OAM mean score | Replay mean score | OAM threshold passes | Replay threshold passes |
| --- | ---: | ---: | ---: | ---: |
| 1 | 63.8788% | 44.5417% | 15/20 | 10/20 |
| 2 | 54.2500% | 49.2917% | 14/20 | 11/20 |
| 3 | 54.6250% | 53.7500% | 12/20 | 10/20 |
| 4 | 73.0871% | 59.8169% | 16/20 | 14/20 |
| 5 | 52.0000% | 40.7500% | 13/20 | 9/20 |
| 6 | 71.4921% | 61.1111% | 15/20 | 14/20 |
| 7 | 60.9375% | 53.1250% | 14/20 | 13/20 |
| 8 | 58.7917% | 47.2292% | 13/20 | 8/20 |
| 9 | 52.7500% | 43.9583% | 13/20 | 9/20 |
| 10 | 73.7500% | 68.0000% | 17/20 | 14/20 |

## 2. Configuration and evidence

| Setting | Recorded campaign |
| --- | --- |
| Dataset | BEAM-10M, ten conversations and 200 distinct questions |
| Dataset revision | `9b2096193fe74e2837e4713e483351e19817773c` |
| Dataset SHA-256 | `c16643a7cc7d93c1ed5509568cdd9e4c387dcd60414e68dde740b54064e68ba8` |
| OAM path | Native L1 indexing and normal recall, low extraction reasoning, Profile disabled, L2 absent from scored contexts |
| Embedding/tokenization | Qwen3 embedding with a 32K-token input limit; approximate OAM tokenizer during evaluation |
| Recorded answer settings | DeepSeek Flash, high and max as recorded per selected answer |
| Judge | DeepSeek Flash, high reasoning |
| Dates | Source batch reference dates; retained next-batch date imputation for the first undated batch of conversation 7 |

All selected answers, scores, contexts and actual request settings remain unchanged. The [failure review](beam10m-failure-review.md) retains the 58-case source-assisted classifications: 30 ambiguous gold answers, two incorrect gold answers, 14 retrieval misses, one answer-handling error and 11 unresolved. Reference issues account for 32/58 failures (55.2%) under that review. Official scores and the denominator are unchanged. This public summary reuses existing review findings; it is not a new audit of all original source conversations.

All 200 OAM contexts have retained approximate noise assessments. Of 58 failed-question contexts, 57 have canonical exact source-assisted noise ratios; one remains uncertain. Unavailable historical stages and incomplete billing are limitations, never assumed successes or zeros. The saved final-summary hash is `40564e93d22a316ba77d7763bb25a345d54af1ceafa1853be6d9ca982779e9f5`; the saved outcome-selection snapshot hash is `1c2da27f3c9677223ea0b41641203c90ce70b776b568a6e89561bb2a2db60c99`.

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

### 3.3 Retained answer/judge usage

Published completed answer/judge receipts total 1,001 calls and 9,602,485 generation tokens for OAM, including 769,550 reasoning tokens within output, and 981 calls and 8,072,232 tokens for the replay, including 632,995 reasoning tokens within output. These totals cover all retained completed receipts, rather than only the selected answers. Generation tokens include input and output; cached input is included in input. Indexing, embeddings, reviews and failed or unknown attempts outside these receipts are excluded. This is not total campaign billing.

## 4. Public per-question evidence

The [manifest](manifest.json) lists ten compressed conversation shards per arm. Each record preserves the original question, gold and rubric, exact saved context and SHA-256, `cl100k_base` context count, saved retrieval timing, recorded answer outputs, selected score and actual model settings. The report displays only the selected saved answer and its judgments. The frozen source files also retain additional recorded outputs needed to inspect the original measurements; they do not define the current evaluation workflow.

Saved request/response projections preserve visible content, usage, finish reasons and original receipt hashes. Ordered rubric judgments retain each criterion, score and reason. Averaging normalized rubric scores reproduces every selected score. Missing runtime revision metadata remains null. Source receipt hashes identify original artifacts rather than projected public JSON.

The data retain [BEAM attribution and CC BY-SA 4.0 terms](LICENSES/BEAM-CC-BY-SA-4.0.md), including applicable separate prompt notices. Two metadata-adjudicated records retain schema-invalid visible responses and recorded answer projections with unchanged answer text. Full source histories, credentials, endpoints, transport headers, hidden provider reasoning and operational logs are excluded. Visible reasoning requested by the benchmark remains in the saved response evidence. These records permit inspection and rescoring but do not recreate indexing or certify a fresh matched run.

Run `uv run pytest -q tests/test_published_oam_evidence.py` to verify coverage, selected scores, threshold counts and exact context tokens. Python's `gzip` and `json` modules open the shards without provider credentials. Rebuild the offline report with `PYTHONPATH=src uv run python scripts/build_historical_reports.py eval_results/beam10m`.
