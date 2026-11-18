# LME-500: 476/500 with MiniMax adaptive at 18k

## Overview

OAM records **476/500 (95.2%)** on the unchanged original LongMemEval S selection: **hard100 83/100** and **remaining400 393/400**. MiniMax M3.1 Flash uses native adaptive thinking for both answering and judging. Normal recall has an **18,000-token context limit**, with **11,373.58 mean saved context tokens** across 497 scored cases. Three disputed-gold cases remain incorrect in the original 500-question denominator.

[Open the offline question browser](report.html) to inspect every outcome and all current saved contexts, visible answers and judgments. This bundle is a projection of completed evidence; publishing it makes no new model, retrieval or indexing calls. The earlier [466/463 same-context comparison](../lme500/README.md) remains unchanged.

## 1. Score and evidence coverage

| Cohort | Correct | Incorrect | Current scored cases | Carried No |
| --- | --- | --- | --- | --- |
| Hard 100 | 83 | 17 | 97 | 3 |
| Remaining 400 | 393 | 7 | 400 | 0 |
| Full selection | **476** | **24** | **497** | **3** |

All 500 canonical IDs occur once and retain their original questions, types and fixed gold answers. `0a995998`, `dd2973ad` and `370a8ff4` have `evaluation_status: carried_incorrect_gold_no`, score zero and no current context, answer, judge or timing. They are included in score/category/filter denominators. Context and retrieval statistics use the 497 measured cases instead of inventing three empty measurements. No question is left unrun under this policy.

The first usable explicit boolean judge verdict is retained, including inline `correct=false`. A valid No is not retried to obtain a pass. The shards retain 994 accepted role responses and 21 saved failed/unusable attempts. These 21 attempts are not additional accepted answers. Failed calls and usage remain distinguishable from accepted calls; missing usage is unavailable, not zero.

## 2. Evaluated source and retrieval settings

The [manifest](manifest.json) pins the **final OAM main commit** [`38bd45d82f24aa33932a7d4b7afdf6ee27585f72`](https://github.com/rocke2020/open-agent-memory/commit/38bd45d82f24aa33932a7d4b7afdf6ee27585f72) published for this result. It was committed after the evaluation; all **116 frozen runtime file hashes** in [recall-source-hashes.json](recall-source-hashes.json) match that commit. This establishes source equivalence for those runtime files, without claiming the final merge SHA existed when the calls ran. No fresh v0.0.3 release-image measurement is inferred. Existing completed L1 indexes were reused; indexing was not repeated.

The [recorded retrieval override](retrieval.yaml) contains the exact non-secret retrieval settings from the frozen runtime YAML. Apply it to the pinned OAM canonical configuration and request `recall_level: normal` with `include.chunks: true`. It is an evaluation override, not a complete standalone server configuration. The product's normal context default is 24,000 tokens; this campaign overrides it to 18,000.

| Setting | Recorded value |
| --- | --- |
| Recall / representation | Normal; L1; complete required source bodies; chunks included |
| Ranking | Complete original query; whole-query lexical ranking; RRF `k=60` |
| Local evidence | `user_fact_exchange`; adjacent user context enabled, 1,000-character bound |
| Candidate caps | 64 per channel; 128 fused; 36 source seeds; 96 learned items and 96 original bodies |
| Optional paths | Entity/semantic graph, causal graph, Insights, Profile and reranker disabled |
| Inactive setting | `rerank_candidates: 48` has no effect because `models.reranker: null` |
| Deadline / context counter | 5 seconds; `o200k_base.encode_ordinary` |
| Answer / judge | `minimax-m3.1-flash` for both, native adaptive |
| Wire options | `thinking: {type: adaptive}`, `reasoning_split: true`, no explicit `reasoning_effort`, no structured-output mode |
| Prompt source | Exact AMB answer and category-specific judge prompts at `7021c59c3497d1dd08d61f12fa4629be983fcfb1` |

Prompt source: [LongMemEval dataset implementation](https://github.com/vectorize-io/agent-memory-benchmark/blob/7021c59c3497d1dd08d61f12fa4629be983fcfb1/src/memory_bench/dataset/longmemeval.py). Complete private receipts were checked against the pinned prompt construction before publication. AMB has no explicit repository license at this revision, so this bundle omits request-message expression and separate model reasoning text. It retains canonical-JSON SHA-256 hashes of request messages, original receipt hashes, non-prompt request options, complete visible outputs, finish reasons and reported usage. It does not relicense AMB prompt text.

The configured route was Tencent Cloud AI. Endpoint addresses and credentials are omitted. Model names and reported response names are preserved as recorded; they are not normalized.

## 3. Context, latency and recorded Hindsight comparison

| Measurement | OAM |
| --- | --- |
| Complete saved context tokens | **5,652,671** across 497 cases |
| Mean / maximum context | **11,373.58 / 18,000** |
| Raw retrieval HTTP samples | **994**, two per scored question; no exclusions |
| Retrieval mean / median / p95 | **332.95 / 328.59 / 428.25 ms** |

Retrieval HTTP timings include ordinary embedding time and exclude answer/judge generation. The median of per-question arithmetic means is 327.94 ms; it differs from the median of 994 raw samples. The report uses the raw-sample median. Saved contexts remain the exact texts scored, independent of timing aggregation.

| Recorded pipeline | Accepted answers | Mean saved formatted context, `o200k_base` |
| --- | --- | --- |
| OAM, MiniMax M3.1 Flash answer and judge | **476/500 (95.2%)** | **11,373.58**, 497 contexts |
| Hindsight, Gemini 3.1 Pro Preview answer and Gemini 2.5 Flash Lite judge | **473/500 (94.6%)** | **43,124.55**, 500 contexts |

OAM's recorded score is 0.6 percentage points higher, with **73.63% fewer saved formatted context tokens**. The comparison is descriptive: answer/judge models, context populations and complete pipelines differ. The [AMB Hindsight artifact](https://github.com/vectorize-io/agent-memory-benchmark/blob/7021c59c3497d1dd08d61f12fa4629be983fcfb1/outputs/longmemeval/hindsight/rag/s.json.gz) has compressed SHA-256 `d0bcc75fb4060f0a395239a353d2066d3f7d252bbe16d674c951d9855deaf9cf`; its actual server revision and complete resolved configuration are not established. Hindsight's answer path can use raw recall JSON rather than the saved formatted string, so these context counts do not establish exact model-input usage or billing savings.

MiniMax's official vendor list price is **USD 0.15 per million input tokens, USD 0.60 per million output tokens and USD 0.015 per million cached input tokens**, checked November 17, 2026. This supports describing it as a low-priced model, without asserting a worldwide cheapest-model ranking. Actual Tencent Cloud AI campaign billing has not been reconciled. [Official MiniMax pricing](https://platform.minimax.io/docs/pricing/overview).

## 4. Improvement and interpretation

Against the earlier MiniMax **466/500** arm, the same 500 IDs have 454 correct in both, 22 newly correct, 12 newly incorrect and 12 incorrect in both: a net gain of **10 answers, or 2 percentage points**. The 12 newly incorrect include one carried disputed-gold No, not a fresh model failure. Retrieval, formatting, prompts and judge model changed together; this is not an isolated estimate of retrieval's causal contribution.

The retrieval work preserves complete original evidence, removes redundant assistant facts and shared-source duplication, improves local source eligibility and temporal presentation, and reduces formatting overhead. On identical selected evidence, temporal-format compaction reduces total context tokens from 5,984,906 to 5,652,671: **332,235 tokens (5.55%)** saved while retaining **4,137,965 evidence-body tokens**. This is a formatting measurement; complete semantic relevance and source-to-context audits remain partial.

The accepted 95.2% result is a fixed-reference judge-acceptance score. Indexing generation usage/time, complete campaign usage, final monetary cost and an engine-only matched comparison are unavailable. Embedding tokens are excluded from generation totals; output usage includes reasoning where reported. The saved failed-attempt receipts prevent treating failures as free.

### Answer and judge variability

Accuracy also depends on answer generation and model-based judging. In a [saved same-input MiniMax example](../model-variability-20261118/README.md), the identical final context and complete answer-request settings produce a correct 27:45 answer in one attempt and an incorrect 28:00 answer in another. This demonstrates observed answer variation, not its frequency. A selected example or one paired small-cohort rerun does not establish a randomness rate or an error bar. The judge inputs differ because the answers differ, so independent judge variability is not isolated. This separate diagnostic does not change the published 476/500 or hard100 83/100 results.

## 5. Validate saved evidence and source builds

From an OAMB checkout, validation and HTML rebuilding make no model calls:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
uv sync --locked
uv run pytest -q tests/test_published_lme500_20261117.py tests/reporting/test_historical_report.py
uv run python scripts/build_historical_reports.py eval_results/lme500-20261117
```

Build OAM from the exact final main commit in the manifest using the [source-image instructions](../../README.md#oam-development-image). No local uncommitted product files enter that build. The build itself and a new full evaluation were not performed for this evidence publication.

The current [native execution workflow](../../docs/native-oam.md) uses the earlier LME fast-recall and prompt protocol. It remains available for separate new runs, but changing only its answer/judge model settings does not execute this normal-recall/AMB-prompt campaign. This bundle supports offline inspection and score recomputation; it does not claim a turnkey native replay of the latest protocol.

## 6. Provenance and redistribution

The cleaned LongMemEval S dataset revision is `98d7416c24c778c2fee6e6f3006e7a073259d48f`, SHA-256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`. Question and context evidence retain the [LongMemEval MIT notice](LICENSES/LongMemEval-MIT.txt). This public projection omits private paths, endpoint addresses, keys and separate reasoning text. Original source receipts remain unchanged; each active row retains their hashes, and each shard hash is recorded in the manifest.
