# Hindsight Implementation Comparison: OAMB and the Reference AMB Checkout

## Overview

1. At the mainflow level, the current OAMB Hindsight path and the reference AMB path are mostly the same: one isolated history per LongMemEval question, Hindsight retain and recall, a separate answer model, and a category-sensitive judge.
2. OAMB uses a digest-pinned Hindsight 0.9.2 REST service, create-only banks, strict input and response validation, the API-default `mid` recall level, the first configured number of provider-ranked normalized facts-and-chunks, explicit model-role bindings, and independently reconstructable evidence. The reference AMB checkout uses an embedded Hindsight 0.4.x daemon, delete-and-recreate banks, an explicit `high` recall level, raw recall JSON in the answer prompt, environment-selected models, and a compact score-oriented result file.
3. Accuracy is the headline difference: the saved OAMB Hindsight LME-60 result is 57/60, or 95.00%, while the published reference AMB result is 473/500, or 94.60%. OAMB is 0.40 percentage points higher by arithmetic comparison, but the differing question sets and end-to-end protocols prevent interpreting that difference as a Hindsight-only effect.
4. The implementation comparison was traced from OAMB base commit `fb35c8a38a85b67a2af2c9e6928c72e3c7aadb4d` and reference AMB commit `f0edfb9fe44ebd1ec9bba8e1737319a29ee0d696`. The OAMB accuracy comes from the later saved 60-record result map with SHA-256 `e63576615d3b8a31e8ea112e8025181db1628c3353cd5d4709770b92bfe45d3f`; the AMB accuracy comes from the pinned 500-record results manifest. Neither result is presented as a reproduction of the other.

## 1. Scope and terminology

The comparison is intentionally narrow: OAMB's active LongMemEval Hindsight REST cell is compared with the reference checkout's default `--dataset longmemeval --split s --memory hindsight --mode rag` path. AMB's optional hosted and self-hosted HTTP providers, `agent`, `agentic-rag`, oracle, and other datasets are outside the behavioral baseline except where they clarify that the default `hindsight` provider is the embedded implementation.

The reference snapshot is a later DeepSeek-provider development descendant of the first public AMB result snapshot, not that historical snapshot itself. Its Hindsight adapter, runner, and model clients have changed, so even an exact run of the inspected checkout would be a current-reference run rather than a strict recreation of the published 473/500 result.

| Term | Meaning in this document |
|---|---|
| OAMB | The implementation in this repository at the pinned commit above. |
| Reference AMB | The separately inspected Agent Memory Benchmark checkout at the pinned commit above. |
| History | The complete set of LongMemEval sessions associated with one question. Both implementations isolate this unit into one Hindsight bank. |
| Native candidate | One fact or hydrated source chunk returned by Hindsight before the benchmark formats answer context. |
| Visible evidence | The exact provider-derived bytes OAMB sends to its answer model. |
| Retrieval generation | A generative call inside memory retrieval, such as Hindsight `reflect`. Ordinary query embedding is not retrieval generation. |

## 2. Shared semantic core

The ingestion, retrieval, answer, and judge stages have the same responsibility boundaries, while their concrete request and evidence contracts differ. Both paths:

1. Target the cleaned LongMemEval S dataset family and represent each conversation session as one Hindsight document.
2. Use the question as the isolation unit, so one question's history is not mixed with another question's recall scope.
3. Remove the dataset-only `has_answer` marker before ingestion and keep the gold answer out of retain, recall, and answer generation.
4. Disable Hindsight observations when creating a bank.
5. Pass the question timestamp to Hindsight recall and include the question date in the LongMemEval answer prompt.
6. Use RAG as `retain history → recall once → answer with a separate LLM → judge with a separate LLM`; neither RAG path invokes Hindsight `reflect`.
7. Apply different judge instructions for ordinary, temporal, knowledge-update, and preference questions, with gold data entering only at the judge stage.

These shared choices do not make the runs interchangeable because the bytes retained, Hindsight runtime, recall controls, evidence passed to the answer model, and output contracts still differ.

| Comparison layer | Verdict |
|---|---|
| Stage graph and component ownership | Mostly the same. |
| Ingestion intent | Mostly the same: one question bank, one document per session, observations disabled. |
| Recall intent | Mostly the same: one timestamp-aware `recall` with chunks and no `reflect`. |
| Answer/judge intent | Mostly the same: separate answer and judge roles, with gold data visible only to the judge. |
| Executable requests, context bytes, failure semantics, and evidence | Materially different. |
| Score protocol | Not equivalent without additional controls. |

## 3. Concise differences

Accuracy is the largest visible outcome difference: OAMB reports 95.00% and reference AMB reports 94.60%, an arithmetic difference of 0.40 percentage points in OAMB's favor. This compares 60 balanced questions with 500 questions under different Hindsight versions, retrieval settings, answer inputs, models, prompts, and completion rules, so it is descriptive rather than controlled.

The implementations share a semantic stage graph, but they are not protocol-equivalent benchmark cells. The most important differences are the evidence shown to the answer model, the Hindsight retrieval/runtime configuration that produces it, and the rules that decide whether a question is complete and scoreable.

| Priority | Comparison point | OAMB | Reference AMB | Why it matters |
|---:|---|---|---|---|
| 1 | Accuracy | Saved OAMB result map: 57/60, or 95.00%, with 60 unique terminal judged records. | Published AMB results manifest: 473/500, or 94.60%. | OAMB is 0.40 percentage points higher arithmetically; different samples and protocols prevent a Hindsight-only attribution. |
| 2 | Answer-visible evidence | Sends the first configured number of provider-ranked normalized facts and hydrated chunks as deterministic compact `F#`/`S#` blocks. | Sends serialized raw recall JSON; its saved formatted context is not the actual answer context. | The answer models reason over substantially different content and markup even if recall returned the same native candidates. |
| 3 | Recall breadth | Uses the Hindsight 0.9.2 API-default `mid` level, 4,096 fact tokens, and 8,192 chunk tokens, then keeps the first configured 150 normalized candidates. | Explicitly uses `high` with 32,768 fact tokens and 32,768 chunk tokens; the adapter does not forward its nominal `k=10`. | Search work, returned evidence volume, and truncation behavior are not aligned. |
| 4 | Hindsight runtime | Uses a digest-pinned Hindsight 0.9.2 REST service with pinned embedding configuration and native reranking disabled. | Uses an embedded 0.4.x daemon profile whose embedding and cross-encoder choices remain daemon-owned. | Extraction, indexing, retrieval, response shape, and ranking come from different Hindsight generations and configurations. |
| 5 | Dataset and retained input | Uses the pinned balanced LME-60 selection, validates the source file and schema, canonicalizes chronology, and retains compact role/content JSON. | Loads all 500 questions by default from an environment path or unpinned download, preserves source order, and retains default `json.dumps` bytes. | Question membership, ordering, validation, and extraction input bytes differ. |
| 6 | Bank lifecycle and readiness | Creates a fresh bank per ingestion occurrence without deleting or reusing provider state, then reconstructs and validates the complete provider projection before recall. | Deletes and recreates a deterministic per-question bank, then proceeds after retain returns or a failed batch is logged and skipped. | Freshness, destructive behavior, and completeness guarantees differ. |
| 7 | Answer and judge protocol | Uses OAMB-authored answer instructions, plain-text answers, exact `yes`/`no` judge output, and artifact-bound model roles and attempts. | Uses AMB prompts, structured reasoning plus answer, a structured judge verdict/reason, and environment-selected model clients. | Generation, parsing, correction, retry, and judging can change verdicts independently of retrieval. |
| 8 | Failure and completeness semantics | Records intended, accepted, and skipped sources and preserves attempts and partial evidence. | Logs and may skip failed retain batches without representing partial ingestion as structured result state. | The two runners can include different histories in scored results and make different completeness claims. |
| 9 | Result purpose | Produces independently validated capsules with requests, raw receipts, projections, attempts, accounting, and a report-facing result subset. | Produces a compact score-oriented JSON result with answer, context, raw recall, verdict, timing, and aggregate accuracy. | The formats support different evidence claims even when both contain an accuracy number. |

The practical conclusion is that a score delta is a combined pipeline difference, not evidence of a Hindsight version improvement or regression. An apples-to-apples experiment would need to freeze the dataset, retained bytes, runtime and bank configuration, recall request and normalization, answer-visible context, answer and judge models/prompts, retry behavior, and score denominator.

## 4. End-to-end flows

The two flows place responsibility at different boundaries. OAMB freezes and validates each boundary before consuming it, while reference AMB keeps most behavior inside one CLI process and one mutable result file.

### 4.1 OAMB

```text
comparison config
  → doctor emits an immutable resolved plan
  → verify the pinned Hindsight service and model-role bindings
  → allocate a fresh create-only bank for one question history
  → sort sessions canonically and retain them one at a time
  → reconstruct the complete provider projection and mark ready
  → project before recall
  → one generation-free Hindsight recall
  → project after recall and reject protected-state mutation
  → normalize returned facts and hydrated source chunks, then keep the configured first N
  → render exact compact visible evidence
  → answer model
  → category-sensitive judge model
  → seal attempts, raw receipts, accounting, and the case result
  → atomically publish the ordinary question result
  → validate the capsule and build the comparison report
```

### 4.2 Reference AMB

```text
CLI plus environment
  → load LongMemEval queries and session documents
  → start or reuse the embedded Hindsight daemon profile
  → delete and recreate the deterministic bank for one question
  → retain source-order sessions in ordered batches
  → one Hindsight recall with a large explicit token budget
  → build formatted documents and also retain the raw recall object
  → serialize raw recall JSON into the LongMemEval answer prompt
  → answer model returns structured reasoning plus answer
  → category-sensitive judge returns structured reason plus verdict
  → write the question into one JSON result file
```

Reference AMB has no resolved-plan, provider-projection, raw-attempt capsule, or independent validation stage. Its result is directly useful as a benchmark summary, but it does not prove the same properties as an OAMB capsule.

## 5. Detailed implementation comparison

The largest differences are outcome-affecting protocol choices, not naming or packaging differences.

| Dimension | OAMB current implementation | Reference AMB current implementation | Practical effect |
|---|---|---|---|
| Hindsight runtime | REST API 0.9.2, source revision and container digest pinned; pg0 state is external to the benchmark package. | Default `hindsight` selects `HindsightEmbedded`; the lock resolves `hindsight-all` and client/embed 0.4.17 while the in-process API package is 0.4.15, and a separate daemon is launched through `uvx`. | Extraction, indexing, retrieval, response shape, and defaults are from different Hindsight generations. |
| Embedding and reranking | Pins `qwen3-embedding:0.6b`, dimension 1024, and disables native reranking; the bank config is read back and matched exactly. | Delegates embedding and cross-encoder choices to the embedded 0.4.x daemon profile; the adapter does not disable the daemon's native reranker. | Candidate production and ordering are not aligned. |
| Dataset scope | Pins the input revision, SHA-256, byte count, exact balanced LME-60 IDs, order, and six-category counts. | Loads all 500 questions by default from an environment path or an unpinned Hugging Face `main` download; category, query, and document limits are CLI options. | OAMB's 60-question study is a screen, not the same sample or denominator as the 500-question AMB result. |
| Input validation | Rejects a changed file hash/size, unknown fields, invalid timestamps or weekdays, misaligned arrays, and duplicate question IDs. | Parses dates permissively, returns `None` on failure, and truncates misaligned session/date/ID arrays to their shortest length. | OAMB fails before evaluation; AMB may continue with a reduced or less precisely dated history. |
| Question bank identity | Derives a bank ID from the unique ingestion occurrence, inventories existing banks, rejects collision or replay, creates once, and never deletes or reuses it. | Uses `{dataset}-{split}-u{question_id}` inside a daemon profile; every pending unit first tries to delete that bank, ignores every delete error, and then recreates it. CLI run name does not enter backend bank identity. | OAMB preserves old provider state and makes each attempt fresh; AMB favors repeatable reruns at a stable name but destructively replaces that bank. |
| Session document bytes | Uses compact UTF-8 JSON with only `role` and `content`; document identity is a hash bound to workload, question, session, ordinal, and payload. | Uses default `json.dumps` output and document ID `{question_id}_{session_id}`; repeated document IDs are deduplicated by keeping the first. | The extraction model receives different bytes, context strings, and document IDs even over the same conversations. |
| Chronology | Parses every timestamp and stably sorts sessions by canonical time and original ordinal; message order within a session is unchanged. | Preserves the source array order and describes it as dataset time order, but does not sort or verify chronology. | They match only when the source is already correctly ordered; OAMB enforces the invariant. |
| Retain granularity | One synchronous REST retain per session, strictly sequential within a history. | Embedded path uses synchronous-completion batches of up to eight sessions, sequential within a bank; multiple banks may overlap. | Extraction batching and failure blast radius differ. |
| Retain failure | Retries only a closed set of independently classified, settled transient failures; after three harness submissions, the affected source is explicitly recorded as skipped and the history becomes partial. Unknown outcomes fail closed. | Retries each batch up to three times, but timeouts, duplicate/FK/empty/connection errors and ultimately any repeated exception can be logged and skipped. Retain responses and extraction usage are discarded. | Both may continue after a missing batch, but only OAMB's result closes intended, accepted, and skipped source counts with retained evidence. |
| Readiness | Reconstructs a complete paginated document/memory projection, verifies original text and fact state, proves no observations or mental models, then seals the ready state. | Treats successful return—or a logged-and-skipped retain—as sufficient to proceed; there is no independent inventory check. | A nonempty AMB result does not prove complete ingestion. |
| Recall request | Sends the full question, `types=[world, experience]`, question timestamp, trace, chunks, and no entities. It omits `budget` and `max_tokens`, and sends an empty chunk-options object, so the Hindsight 0.9.2 HTTP defaults apply: `mid`, 4,096 fact tokens, and 8,192 chunk tokens. | Truncates the question to 1,900 characters and explicitly requests `budget=high`, 32,768 fact tokens, 32,768 chunk tokens, chunks, no entities, and the question timestamp; it does not restrict fact types. | The reference path searches with a larger thinking budget and can expose far more evidence to the answer model. |
| Top-k semantics | `retrieval.top_k=150` is frozen from `benchmark.yml`; Hindsight has no item-count request field, so OAMB keeps the first 150 provider-ranked normalized facts and chunks. | The provider interface defaults to `k=10`, but the Hindsight adapter does not forward it. | OAMB has an answer-visible normalized-candidate ceiling, while provider-side search breadth and response volume still come from Hindsight's token budgets and defaults. |
| Retrieval generation | The exact outbound request is sealed and independently checked as one `recall` with no `reflect`; reranking is disabled. | RAG calls `recall` once and does not call `reflect`, but this is inferred from the selected mode and source flow rather than sealed request proof. | Both RAG paths are generation-free inside retrieval, but only OAMB makes it an artifact-level validation claim. |
| Recall normalization | Validates a closed 0.9.2 response shape, rejects unknown documents, duplicate fact IDs, unexpected fact types, entities, source facts, and reranker scores, then preserves fact order and adds every hydrated chunk as a separately identified candidate. | Deduplicates formatted results by `chunk_id` or fact ID and inlines a chunk only on its first formatted result; it does not validate provider identities against the requested bank. | OAMB makes facts/chunks independently countable; AMB's formatted view is more compact but less strictly attributable. |
| Actual answer context | Keeps the configured first 150 provider-ranked normalized facts and chunks, then removes duplicate evidence identities; it has no separate character or token ceiling and renders deterministic `F#`/`S#` blocks with timestamps, source links, and native truncation markers. | Although it builds formatted documents, LongMemEval substitutes `json.dumps(raw_recall_response)` into the actual answer prompt whenever raw response exists. The saved `context` remains the formatted/deduplicated view. | The answer model sees substantially different content and markup; AMB's saved context is not the exact text used in its answer prompt. |
| Answer prompt and output | Uses an OAMB-authored concise evidence prompt with explicit counting, latest-update, preference, unavailable-data, question-date, and native-truncation semantics; accepts one nonblank plain-text answer with a normal finish. | Uses a longer AMB LongMemEval prompt that asks for comprehensive detail and has different recommendation and ambiguity instructions; requires structured `reasoning` plus `answer`. | Answer behavior is deliberately different even with identical retrieval and model. |
| Judge | Uses attributed LongMemEval category templates and requires the complete output to normalize to exactly `yes` or `no`; it stores a binary numerator and denominator. | Uses category-sensitive AMB templates and a structured `correct: boolean` plus `reason: string` response. | The criteria overlap, but prompt bytes, response contract, parsing, and correction paths differ. |
| Model binding | Resolved plan binds extraction, embedding, answer, and judge roles, endpoints, model names, ownership, and thinking effort; runtime evidence checks those bindings. Tracked config intentionally leaves concrete light/deep model strings to the environment. | Extraction, answer, and judge are independently selected through environment variables; the result records configured answer/judge IDs but not extraction, embedding, reranker, reasoning effort, effective runtime model, or usage. | A source checkout alone does not prove either run's concrete external models, but OAMB is designed to close them in the run artifact. |
| Model retries | Answer and judge each allow six outer attempts and two transport retries per outer attempt, with separate evidence for every physical call and correction prompts for invalid output. | Retries live inside each LLM client. The OpenAI path shares six attempts across rate errors and DeepSeek JSON correction; the Gemini structured-output path can nest two six-attempt loops. | Retry count and model conversation history can differ, affecting both cost and outputs. |
| Parallelism | Current checked-in limits are three provider cells, three histories per provider, and three question pipelines per provider. A Hindsight-only run can overlap three histories and three retrieve→answer→judge pipelines; all three providers can theoretically reach nine of each. Sources within one history remain serial. | One CLI process evaluates one provider. It prefetches the current plus four future bank ingestions, while LongMemEval's one-query-per-unit loop makes question pipelines effectively serial. | OAMB overlaps complete question work more aggressively; reference AMB overlaps more bank ingestion chains. |
| Shutdown and partial failure | Stops admission, drains admitted operations, closes clients after active calls settle, and preserves aborted or partial evidence. | A recall, answer, or judge exception aborts the runner; there is no explicit drain of prefetched ingestion tasks and cleanup is reached only after the final save. | OAMB has stronger interrupted-run semantics; AMB relies primarily on per-unit checkpoints. |
| Results and metrics | Seals requests, raw responses, prompts, projections, attempts, latency, token/resource/cost records, and native/visible candidate counts; ordinary result files retain the report-facing subset. | Stores answer, reasoning, formatted context, raw recall object, verdict/reason, recall time, aggregate accuracy, and local context tokens, but no attempt ledger, extraction/model usage, cost, or completeness proof. | The two outputs answer different evidence questions even when they both contain an accuracy number. |

### 5.1 Hindsight recall levels and token limits

Hindsight's `low`, `mid`, and `high` recall levels control retrieval work breadth through a separate integer thinking budget; they are not LLM reasoning tokens, returned text tokens, or a minimum amount of content. Under the fixed mappings used by these versions, `low`, `mid`, and `high` map to 100, 300, and 1,000 respectively. Hindsight passes the resolved number as the limit for its unified semantic, BM25, graph, and temporal retrieval work, then applies an independent candidate prefilter before retaining at most twice the thinking budget for token filtering. The pinned runtime's default prefilter is 300 candidates, so OAMB's `thinking_budget=300` has a theoretical later window of 600 but no more than 300 candidates reach it under this configuration; the actual returned fact count may be much lower.

| Path | Selected level | Effective thinking budget | Fact result limit | Chunk hydration limit |
|---|---:|---:|---:|---:|
| Reference AMB | Explicit `high` | 1,000 | 32,768 tokens | 32,768 tokens |
| OAMB | Implicit API-default `mid` | 300 | 4,096 tokens | 8,192 tokens |

OAMB separately validates bank configuration fields `recall_max_tokens=2048` and `recall_chunks_max_tokens=1000`, but those are not the effective limits of this direct REST call. The request model supplies 4,096 when `max_tokens` is omitted and 8,192 when `include.chunks={}` has no `max_tokens`; the HTTP route passes those request-model values to the retrieval engine. This source-to-runtime distinction is why the OAMB row above uses 4,096 and 8,192 rather than the bank configuration values.

The controls are independent: the per-request `budget` selects `low`, `mid`, or `high`; bank configuration maps that level to a fixed or adaptive numeric thinking budget; request `max_tokens` limits returned fact text; and `include.chunks.max_tokens` limits hydrated source-chunk text. OAMB currently hard-codes omission of `budget` and the two token values rather than exposing them in `benchmark.yml`. Selecting another level therefore requires a deliberate adapter request change plus matching request-proof validation and tests, not only a configuration-file edit.

### 5.2 Context-token accounting is not comparable

OAMB counts the exact compact visible-evidence bytes with a pinned `tiktoken` 0.14.0 `o200k_base` fingerprint. Reference AMB counts its saved formatted context with `cl100k_base`, while the LongMemEval answer prompt normally receives serialized raw recall JSON instead. Reference `context_tokens` is therefore neither the same tokenizer nor the token count of the actual Hindsight answer context.

### 5.3 Ingestion counters are not comparable

OAMB distinguishes intended, accepted, and skipped source units and binds extraction usage to physical retain attempts. Reference `ingested_docs` counts source document occurrences presented by the runner; it does not account for duplicate-ID removal, skipped batches, extracted facts, or confirmed provider writes.

## 6. Consequences for benchmark interpretation

The two implementations can be compared as benchmark-system designs, but their accuracy values are not a controlled Hindsight-only experiment.

### 6.1 Claims that are supported

1. Both implement a one-bank-per-question LongMemEval RAG evaluation with observations disabled, a question-time recall, a separate answer model, and a category-sensitive judge.
2. OAMB provides stronger freshness, completeness, request, mutation, retry, and accounting evidence than the reference AMB result format.
3. Reference AMB explicitly uses `high` recall with 32,768-token fact and chunk limits and sends the raw recall object to the answer model, while OAMB implicitly uses the 0.9.2 API-default `mid` level with 4,096 fact tokens, 8,192 chunk tokens, and a compact facts-and-chunks projection.
4. Each implementation can compare Hindsight with other providers only within its own frozen harness, model stack, dataset selection, and completion rules.

### 6.2 Claims that are not supported

1. The arithmetic 0.40 percentage-point difference is not a reproduction, subset estimate, or controlled comparison with AMB's published 500-question 94.60% result.
2. A score delta between the two paths cannot be labeled a Hindsight version improvement or regression because embedding, reranking, retained bytes, recall volume, answer context, prompts, models, and judge contracts all move together.
3. Matching the aggregate score would not establish protocol equivalence or a matching per-question verdict vector.
4. A healthy service, nonempty recall, or complete-looking JSON result alone does not establish that every intended session was successfully indexed.

### 6.3 Minimum controls for an apples-to-apples experiment

A future controlled experiment must freeze the following before interpreting a score delta as a Hindsight change:

1. Identical question membership, order, source file hash, session inclusion, session chronology, and exact retained document bytes.
2. Identical Hindsight release/build, embedding model and revision, reranker configuration, bank configuration, and fresh-scope rule.
3. Identical retain batching, completion condition, retry policy, and treatment of skipped or unknown-outcome writes.
4. Identical recall query bytes, fact types, budget, maximum fact/chunk tokens, entity/chunk options, reranking, and response normalization.
5. Identical answer-visible context bytes, answer prompt, answer model, sampling/thinking settings, output contract, and retry conversation.
6. Identical judge prompt, judge model, output contract, retry behavior, unjudged semantics, and accuracy denominator.

Until those controls are closed, the correct label is a combined pipeline comparison.

## 7. Current limitations and retained boundaries

The stronger OAMB evidence model still has explicit current boundaries, and the reference AMB path should be read according to what its result format actually proves.

### 7.1 OAMB boundaries

1. The resolved retrieval configuration sets `top_k=150`. Because Hindsight's recall endpoint has no item-count field, the adapter applies this as a post-normalization prefix over provider-ranked facts and chunks; the provider request itself still uses API-default `mid` with 4,096 fact tokens and 8,192 chunk tokens. Reports may describe a 150 normalized-candidate ceiling, but must not call it provider-side top-k, high-budget recall, or the separately validated 2,048/1,000 bank values.
2. Concrete light and deep model names come from the run environment. Tracked source proves role ownership and low/high thinking-effort policy, while a resolved plan plus runtime receipts are required to prove the actual model names used by a run.
3. Native capsules retain and independently validate the full request/projection/attempt chain. The current full-study comparison, however, is reduced from ordinary per-provider question-result files; its report intentionally marks run ID, source-root hash, validation hash, code revisions, and accounting outside the retained subset as unavailable.
4. The saved Hindsight result map contains 60 terminal judged records and supports the reported 57/60 arithmetic. The saved bundle does not include native capsule validation files, so this document does not claim capsule-level validation for that score.

### 7.2 Reference AMB boundaries

1. The inspected snapshot is a development branch state, not the historical AMB commit that first published 473/500. Current code behavior must not be back-projected onto that historical run.
2. Default Hindsight setup deletes deterministic banks. Running it against valuable provider state requires an intentionally disposable or backed-up scope.
3. Logged skipped retain batches are not represented as structured partial-ingestion state, and the output cannot independently prove that every intended source reached Hindsight.
4. The stored Hindsight context and `context_tokens` do not reconstruct the actual raw-JSON answer prompt. The raw recall object helps diagnosis, but answer/judge requests, responses, usage, retries, and effective provider identities are not a closed evidence chain.

## 8. Hindsight's three recall levels and three independent limits

The short answer is that OAMB currently uses Hindsight's `mid` recall level, whose fixed `thinking_budget` is 300. This does **not** mean Hindsight generates 300 thinking tokens, returns 300 facts, or always processes more than 300 pieces of content. It is an internal search-breadth control. The two adjacent columns are separate output-text budgets: one packs returned fact text, and the other hydrates raw source-chunk text. Hindsight's official [retrieval guide](https://hindsight.vectorize.io/developer/retrieval) explicitly describes `budget` and `max_tokens` as independent dimensions, and the official [recall API guide](https://hindsight.vectorize.io/developer/api/recall) documents the independent chunk budget and its truncation behavior.

### 8.1 The three levels under OAMB's pinned request and runtime

The following table is specific to OAMB's pinned Hindsight 0.9.2 runtime and exact REST request. The adapter omits `budget` and top-level `max_tokens`, while sending `include.chunks={}`. Therefore the HTTP request-model defaults supply `mid`, 4,096 fact tokens, and 8,192 chunk tokens. The numbers in the last two columns stay constant across the three levels because changing the recall level alone does not change either output budget in fixed mode.

| Recall level | `thinking_budget` | Fact result limit (`max_tokens`) | Chunk hydration limit (`include.chunks.max_tokens`) | Used by OAMB now? |
|---|---:|---:|---:|---|
| `low` | 100 | 4,096 tokens | 8,192 tokens | No |
| `mid` | 300 | 4,096 tokens | 8,192 tokens | Yes, implicitly through the HTTP default |
| `high` | 1,000 | 4,096 tokens | 8,192 tokens | No |

The level values come from Hindsight's fixed mapping: `low → 100`, `mid → 300`, and `high → 1,000`. They are confirmed both by the official [configuration reference](https://hindsight.vectorize.io/developer/configuration#recall-budget-mapping) and by the pinned source's [`_resolve_thinking_budget`](https://github.com/vectorize-io/hindsight/blob/ebad478240d3171bb88201ececda5e8d9883d22d/hindsight-api-slim/hindsight_api/engine/memory_engine.py#L1188-L1219). The 4,096 and 8,192 request defaults come from the pinned [`RecallRequest` and `ChunkIncludeOptions`](https://github.com/vectorize-io/hindsight/blob/ebad478240d3171bb88201ececda5e8d9883d22d/hindsight-api-slim/hindsight_api/api/http.py#L245-L320), not from the recall-level mapping.

For comparison, the reference AMB path's 32,768 fact and chunk limits are caller overrides, not the built-in meaning of `high`. If the same request changed only `budget` from `mid` to `high`, its search breadth would change from 300 to 1,000 while its 4,096/8,192 output budgets would remain unchanged.

### 8.2 `thinking_budget`: search breadth, not model thinking or returned content

`thinking_budget` is Hindsight's internal numeric form of the public `budget` enum. It is unitless and controls how widely the retrieval pipeline searches. It is not an LLM reasoning budget, a prompt-token limit, a completion-token limit, elapsed time, or a guaranteed result count. Ordinary `recall` retrieves and ranks memories; it does not generate an answer with an LLM, as Hindsight's official [recall-versus-reflect explanation](https://hindsight.vectorize.io/blog/2026/07/24/recall-vs-reflect) also clarifies.

In the pinned 0.9.2 code, the resolved number flows into the unified retrieval call as `limit=thinking_budget`. For each requested fact type, the Postgres implementation uses that limit for semantic and BM25 retrieval, temporal retrieval when a temporal constraint exists, and graph traversal as its traversal budget. OAMB requests two fact types, `world` and `experience`, so `thinking_budget=300` is applied per retrieval method and per fact type rather than acting as one global 300-row return limit. The official retrieval guide summarizes the same pipeline-level use, and the pinned [`retrieve_all_fact_types_parallel`](https://github.com/vectorize-io/hindsight/blob/ebad478240d3171bb88201ececda5e8d9883d22d/hindsight-api-slim/hindsight_api/engine/search/retrieval.py#L797-L883) passes the value into the memory store.

The pipeline then merges and deduplicates candidates from the retrieval arms using reciprocal-rank fusion. Before optional cross-encoder reranking, a separate `HINDSIGHT_API_RERANKER_MAX_CANDIDATES` limit caps the fused set; its pinned default is 300 even when cross-encoder reranking is disabled. After scoring, Hindsight takes at most `2 × thinking_budget` candidates into chunk hydration and fact-token selection. For OAMB's current settings, the sequence is therefore `300 per enabled arm and fact type → fused unique candidates → at most 300 after the independent prefilter → at most 600 by the later 2× window`, which means the effective pre-token-filter candidate count cannot exceed 300 here.

```text
public budget: low | mid | high
  → fixed thinking_budget: 100 | 300 | 1000
  → semantic + BM25 + graph + optional temporal retrieval, per fact type
  → fusion and deduplication
  → independent candidate prefilter
  → top 2 × thinking_budget scored candidates
      ├─ hydrate source chunks up to include.chunks.max_tokens
      └─ select fact texts up to top-level max_tokens
```

Consequently, “`thinking_budget=300`” does not imply that any content exceeds 300, that exactly 300 candidates are found, or that 300 facts are returned. The real counts can be lower because the bank may be small, retrieval arms may find fewer matches, the same fact may appear in several arms and be deduplicated, score thresholds may remove candidates, the independent candidate cap may bind, and the fact-token budget may omit otherwise relevant facts. A higher level increases the opportunity to find indirect or lower-ranked evidence and usually costs more search work; it does not force the response to be larger. This is why the official [performance guide](https://hindsight.vectorize.io/developer/performance) recommends `low` for quick lookups, `mid` for standard queries, and `high` for comprehensive queries.

Hindsight can alternatively map the three levels adaptively. In adaptive mode, it computes `round(max_tokens × ratio)` using default ratios 0.025, 0.075, and 0.25 for `low`, `mid`, and `high`, then clamps the result to the configured minimum 20 and maximum 2,000. OAMB pins fixed mode, so its 300 does not grow when fact `max_tokens` grows. Adaptive mode is an operator configuration choice, not the behavior of the current benchmark.

### 8.3 Fact result limit: the fact-text packing budget

The “Fact result limit” column is the request's top-level `max_tokens`. It limits the combined token count of returned facts' `text` fields after retrieval, fusion, scoring, and the candidate-window truncation. It is not a number of facts, and fact metadata such as IDs, types, timestamps, context, tags, and entity references does not spend this budget. The pinned runtime counts these tokens with `cl100k_base`; this is Hindsight's internal packing count, not necessarily the exact token count seen later by OAMB's answer model.

The pinned selection code walks facts in relevance order. A fact that does not fit in the remaining budget is skipped, and selection continues so a shorter lower-ranked fact can still fit. If the query matched candidates but no fact fits at all and `max_tokens` is positive, Hindsight returns the top-ranked fact whole even though the reported fact-token total exceeds the requested limit; this avoids turning “one oversized match” into an apparently empty memory bank. `max_tokens=0` is the explicit exception: it returns no facts and can be used to request chunks alone. These details are documented in the official recall API guide and implemented in the pinned [`select_facts_within_budget`](https://github.com/vectorize-io/hindsight/blob/ebad478240d3171bb88201ececda5e8d9883d22d/hindsight-api-slim/hindsight_api/engine/fact_budget.py#L43-L90).

Therefore 4,096 means “try to pack approximately 4,096 `cl100k_base` tokens of ranked fact text,” not “return 4,096 facts” and not an unconditional hard ceiling in the one-top-fact overflow case. Increasing it can expose more already-ranked facts to the answer model but does not make search deeper in OAMB's fixed-budget mode.

### 8.4 Chunk hydration limit: a separate raw-source-text budget

The “Chunk hydration limit” column is `include.chunks.max_tokens`. A chunk is raw source text from which one or more facts were extracted, so it can provide context or nuance absent from concise fact text. Chunk inclusion is disabled when `include.chunks` is absent or `null`; an empty object enables it with the HTTP default of 8,192 tokens. OAMB sends the empty object and therefore enables chunk hydration at that default.

Chunk hydration is independent of fact packing and occurs first. Hindsight takes the top-scored candidate window, collects unique source chunk IDs in fact-relevance order, and spends the chunk token budget across those raw texts. Complete earlier chunks are returned while they fit. If the next chunk would overflow the remaining budget, Hindsight truncates that final chunk to the remaining token count, marks it `truncated=true`, and stops; it does not skip that chunk and search for a shorter later chunk. The official recall API guide documents this behavior, and the pinned [chunk-hydration pipeline](https://github.com/vectorize-io/hindsight/blob/ebad478240d3171bb88201ececda5e8d9883d22d/hindsight-api-slim/hindsight_api/engine/memory_engine.py#L6965-L7216) performs hydration before fact selection.

Because chunks are chosen before the fact `max_tokens` filter, a chunk can remain in the response even if its associated fact is later omitted, and `max_tokens=0` can still return chunks. OAMB intentionally normalizes any such remaining chunk as evidence rather than discarding it. The 8,192 chunk limit is thus a second context allocation, not part of the 4,096 fact allocation: the provider can return up to roughly 4,096 fact-text tokens plus 8,192 chunk-text tokens, before JSON metadata and OAMB's own evidence formatting.

### 8.5 How to control the three values

For a direct Hindsight REST caller, the three controls are explicit and independent:

```json
{
  "query": "What changed in the deployment plan?",
  "budget": "high",
  "max_tokens": 8192,
  "include": {
    "entities": null,
    "chunks": {
      "max_tokens": 16384
    }
  }
}
```

In this example, `budget="high"` selects the high search-breadth mapping, top-level `max_tokens=8192` controls fact-text packing, and `include.chunks.max_tokens=16384` controls source-chunk hydration. Omitting `budget` selects HTTP-default `mid`; omitting top-level `max_tokens` selects 4,096; using `include.chunks={}` selects 8,192; and omitting `include.chunks` or setting it to `null` disables chunk hydration.

Operators can change how `low`, `mid`, and `high` map to numeric breadth using bank or server recall-budget configuration, including fixed values or adaptive mode. That mapping should not be confused with per-request fact and chunk token limits. Other controls such as `types`, tags, timestamps, minimum scores, enabled retrieval arms, and the independent reranker candidate cap can also reduce or reshape the result set without changing any of these three columns.

OAMB does not currently expose these request values through `benchmark.yml`: its adapter deliberately omits `budget` and fact `max_tokens` and sends empty chunk options, while its request-proof validator expects that exact body. Changing OAMB from the current 300/4,096/8,192 behavior therefore requires changing the adapter request together with the matching validator and tests. Changing only the bank's `recall_max_tokens` or `recall_chunks_max_tokens` values would not alter this direct REST call because the HTTP request model has already supplied its own defaults before invoking the engine.

## 9. Source map

The comparison above was traced from implementation entrypoints rather than inferred from README examples or result labels.

### 9.1 OAMB sources

| Concern | Authoritative source |
|---|---|
| Active dataset, cells, model roles, retrieval policy, concurrency, retry, and timeout configuration | [`configs/benchmark.yml`](../../configs/benchmark.yml), lines 5–191 |
| Pinned provider release/build and retain batch limit | [`src/oamb/config/provider_services.py`](../../src/oamb/config/provider_services.py), lines 54–62; [`provider-services/versions.env`](../../provider-services/versions.env), lines 1–4 |
| Hindsight container, storage, extraction, embedding, reranking, and retry environment | [`provider-services/compose.yaml`](../../provider-services/compose.yaml), lines 4–41 |
| Exact Hindsight bank config and response parsing | [`src/oamb/memory_systems/hindsight/profiles.py`](../../src/oamb/memory_systems/hindsight/profiles.py), lines 19–105 and 155–342 |
| REST request methods | [`src/oamb/memory_systems/hindsight/client.py`](../../src/oamb/memory_systems/hindsight/client.py), lines 42–135 |
| Scope allocation, retain dispatch, readiness, projection, and recall | [`src/oamb/memory_systems/hindsight/adapter.py`](../../src/oamb/memory_systems/hindsight/adapter.py), lines 111–690 |
| Recall result and chunk normalization | [`src/oamb/memory_systems/hindsight/normalize.py`](../../src/oamb/memory_systems/hindsight/normalize.py), lines 12–236 |
| Complete provider projection | [`src/oamb/memory_systems/hindsight/projection.py`](../../src/oamb/memory_systems/hindsight/projection.py), lines 161–542 |
| LongMemEval selection, chronology, source units, answer prompt, and judge | [`src/oamb/workloads/longmemeval.py`](../../src/oamb/workloads/longmemeval.py), lines 223–301, 353–381, 423–465, 523–664, and 688–985 |
| Exact visible-evidence formatting and token counting | [`src/oamb/workloads/visible_evidence.py`](../../src/oamb/workloads/visible_evidence.py), lines 23–194 |
| History/question scheduling, retry, answer, judge, and case sealing | [`src/oamb/runtime/native_run.py`](../../src/oamb/runtime/native_run.py), lines 2087–2267, 2542–2657, 2670–2898, and 2983–3570 |
| Read-only pre/post-recall mutation guard | [`src/oamb/runtime/memory_query.py`](../../src/oamb/runtime/memory_query.py), lines 70–155 |
| Live factories and missing-question selection | [`src/oamb/live.py`](../../src/oamb/live.py), lines 657–868 |

### 9.2 Reference AMB sources at `f0edfb9fe44ebd1ec9bba8e1737319a29ee0d696`

| Concern | Source path in the reference repository |
|---|---|
| Declared `omb` entrypoint and dependency set | `pyproject.toml`, lines 1–30 |
| CLI selection and environment loading | `src/memory_bench/cli.py`, lines 1–72 |
| LongMemEval conversion and answer/judge prompts | `src/memory_bench/dataset/longmemeval.py`, lines 55–175 and 177–354 |
| Hindsight variants, daemon profile, bank lifecycle, retain, recall, and formatting | `src/memory_bench/memory/hindsight.py`, lines 15–216 and 219–830 |
| Embedded-daemon extraction patch | `src/memory_bench/memory/_hindsight_daemon.py`, lines 1–64 |
| RAG retrieval and answer path | `src/memory_bench/modes/rag.py`, lines 29–96 |
| Scheduling, checkpoint, judge dispatch, and result writing | `src/memory_bench/runner.py`, lines 46–444 |
| Published 473/500 Hindsight result | [`results-manifest.json`](https://github.com/vectorize-io/agent-memory-benchmark/blob/decbb07f4f9899deac28a76293564cf263872652/results-manifest.json) |
| Answer and judge model selection | `src/memory_bench/llm/__init__.py`, lines 21–40 |
| OpenAI-compatible structured output and retries | `src/memory_bench/llm/openai.py`, lines 63–139 |
| Gemini structured output and retries | `src/memory_bench/llm/gemini.py`, lines 23–175 |
| Judge wrapper and result contract | `src/memory_bench/judge.py`, lines 27–47; `src/memory_bench/models.py`, lines 24–73 |
| Context-token counter | `src/memory_bench/utils.py`, lines 8–14 |
| Resolved Hindsight package versions | `uv.lock`, lines 1955–2054 |

### 9.3 Hindsight sources used to resolve effective recall defaults

| Runtime | Source evidence |
|---|---|
| OAMB's pinned Hindsight 0.9.2 source revision `ebad478240d3171bb88201ececda5e8d9883d22d` | `hindsight-api-slim/hindsight_api/api/http.py`, lines 245–320 and 4765–4824, defines `budget=mid`, `max_tokens=4096`, and empty chunk options as `max_tokens=8192`; `hindsight-api-slim/hindsight_api/engine/memory_engine.py`, lines 1188–1219, maps fixed `mid` to 300. |
| Reference AMB's Hindsight 0.4.17 API source revision `2191654b1f9b454703916612fec57ce226c7746b` | `hindsight-api/hindsight_api/engine/memory_engine.py`, lines 2339–2342, maps explicit `high` to 1,000; the reference adapter supplies the separate 32,768 fact and chunk limits. |
