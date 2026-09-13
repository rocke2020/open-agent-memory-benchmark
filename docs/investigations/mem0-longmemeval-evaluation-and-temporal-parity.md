# Mem0 LongMemEval Evaluation Comparison: OAMB and Mem0 Managed Evaluation

## Overview

1. We thank the Mem0 maintainers for publishing their LongMemEval runner, prompts, result artifact, and headline configuration. Those sources make it possible to distinguish the shared memory workflow from the parts of each evaluation that affect final system accuracy.
2. Accuracy is the headline difference: the final OAMB Mem0 LME-60 result is 52/60, or 86.67%; the checked Mem0 managed artifact is 467/500, or 93.40%, making OAMB 6.73 percentage points lower by arithmetic comparison. Mem0's current headline is 472/500, or 94.40%, making OAMB 7.73 points lower, but the inspected repository does not publish the corresponding replacement per-question artifact.
3. These percentage-point differences are descriptive, not controlled Mem0 comparisons. OAMB uses a balanced 60-question selection, self-hosted Mem0 2.0.19, `top_k=150`, native reranking disabled, and common OAMB answer and judge roles; the managed result uses 500 questions, managed retrieval, `top_k=200`, GPT-5 answer and judge roles, and Mem0's LongMemEval-specific prompts.
4. The final OAMB result uses the current timestamp-corrected Mem0 path: each original two-message pair retains its source observation time in storage metadata and in a generic New Messages extraction instruction. The 60-question result map is complete, and its retained validation evidence has disposition `validated` with no failed or missing required rule.
5. The checked public managed artifact is complete: 500 unique evaluations, 467 passes, 33 failures, and zero errors. The separate 94.40% README update supplies a numerator and denominator but not matching per-question predictions, judgments, retrieved evidence, runtime configuration, or hashes.

## 1. Scope and terminology

This investigation compares the final OAMB `mem0-rest-v1` balanced LME-60 result with Mem0's published managed LongMemEval evidence. It describes the current evaluated paths, not the history of how either implementation reached them.

| Term | Meaning in this document |
|---|---|
| OAMB Mem0 LME-60 | The final 60-question self-hosted Mem0 2.0.19 result using OAMB's shared provider-comparison protocol. |
| Mem0 managed evaluation | The public `memory-benchmarks` LongMemEval path using Mem0's managed platform and its documented answer and judge prompts. |
| System accuracy | Final judged-answer accuracy of memory ingestion, retrieval, answer generation, and judging together; it is not retrieval recall alone. |
| Descriptive difference | Arithmetic subtraction between reported percentages whose question set or evaluation protocol differs; it does not isolate a provider effect. |

| Evidence | Bound identity |
|---|---|
| OAMB workload | Cleaned LongMemEval S source SHA-256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`; balanced manifest hash `90b2669f7b893e59d404549f5803882bcd6640ce82520a9bf09672cc79464c80`. |
| OAMB result | Saved bundle `mem0-lme60-20260913-oamb/`; `results/mem0.json` SHA-256 `56f60d10db4b5d344430a8e4de8db257ea902b01f70078e7cfd799683de4c045`; resolved-plan hash `17279da8a80b74e0f0ddbc9b2e49a790737f2689ad7afeb9f30617c7b16220f6`. |
| Checked Mem0 artifact | Public `results/platform/longmemeval_results.json`: timestamp `20260406_101243`, 500 evaluations, `top_200`, 467 passed, 33 failed, zero errors. |
| Current Mem0 headline | Public README: 472/500, or 94.40%, for managed `top_k=200`. |

## 2. Shared semantic core

Both evaluations:

1. Use LongMemEval S conversation histories and isolate each question into its own Mem0 scope.
2. Remove dataset-only answer annotations before ingestion and keep the gold answer out of ingestion, retrieval, and answer generation.
3. Ingest chronological user and assistant conversation content into Mem0.
4. Search with the raw LongMemEval question and send retrieved memories to a separate answer model.
5. Use a separate judge that sees the question, gold answer, and generated answer.
6. Report final judged-answer accuracy rather than direct retrieval recall.

These shared stages do not make the scores interchangeable because the dataset selection, Mem0 backend, observation-time translation, retrieval breadth, answer prompt and model, judge prompt and model, and evidence contracts differ.

## 3. Concise differences

Accuracy is the largest visible outcome difference: OAMB is 6.73 percentage points below the checked 93.40% artifact and 7.73 points below the 94.40% headline. The first subtraction compares two retained result artifacts; the second compares OAMB's artifact with a headline whose matching replacement per-question artifact was not published in the inspected repository.

| Comparison point | OAMB Mem0 LME-60 | Mem0 managed evaluation |
|---|---|---|
| Accuracy | 52/60, or 86.67%, with 60 unique terminal judged records and validated result evidence. | Checked artifact: 467/500, or 93.40%; current headline: 472/500, or 94.40%. |
| Question set | Frozen balanced 60: exactly ten questions from each of six types, with source and manifest hashes. | All 500 LongMemEval S questions. |
| Mem0 backend | Self-hosted Mem0 2.0.19 with PostgreSQL/pgvector, v3 additive extraction, and OAMB-pinned dependencies. | Mem0 managed v3 platform; the public result does not fully disclose the deployed retrieval implementation. |
| Ingestion unit | Original chronological user/assistant pairs; role/content bytes remain unchanged. | Consecutive user/assistant pairs under the managed evaluation runner. |
| Observation time | Supplies each source time through `metadata.created_at` and a generic New Messages extraction instruction. | Managed path accepts historical observation time. |
| Retrieval | Provider-native `/search`, `top_k=150`, native reranking disabled. | Managed search with `top_k=200` for the compared artifact and headline. |
| Answer | Common OAMB answer role and provider-neutral prompt shared with Hindsight and OpenViking. | GPT-5 and Mem0's LongMemEval-specific reader prompt. |
| Judge | Common OAMB judge role and attributed category-sensitive LongMemEval rubrics. | GPT-5 and Mem0's evaluation-specific judge prompt. |
| Result evidence | Typed 60-question result map, resolved plan, validation results, and hash manifest. | Checked artifact has all 500 per-question evaluations and aggregate metrics; the newer headline has no matching replacement artifact in the inspected repository. |

## 4. End-to-end flows

### 4.1 OAMB

```text
frozen balanced LME-60 manifest
  → fresh isolated Mem0 run scope for one question history
  → chronological original user/assistant pairs
  → source timestamp in metadata and New Messages extraction instruction
  → self-hosted Mem0 v3 extraction and PostgreSQL/pgvector storage
  → provider-native search(question, top_k=150), native reranking disabled
  → common provider-neutral answer model and prompt
  → common category-sensitive LongMemEval judge
  → atomic question result and independently validated capsule
```

### 4.2 Mem0 managed evaluation

```text
all 500 LongMemEval S questions
  → isolated managed Mem0 scope per question
  → chronological user/assistant pairs
  → managed v3 ingestion and retrieval
  → search(question, top_k=200)
  → GPT-5 with Mem0 LongMemEval reader prompt
  → GPT-5 with Mem0 judge prompt
  → per-question result artifact and aggregate accuracy
```

## 5. Current OAMB temporal translation

OAMB preserves each original two-message pair and maps its canonical `SourceUnit.occurred_at` into two existing Mem0 request inputs:

1. `metadata.created_at` supplies the storage timestamp copied onto extracted memories.
2. The custom extraction instruction states the actual observation timestamp for New Messages, requires relative expressions to resolve against it, and explicitly excludes Last k Messages and Existing Memories from that timestamp.

The instruction contains no LongMemEval question, answer, question type, gold-session identity, or dataset-specific semantic rule. OAMB does not prepend or append the timestamp to the original user or assistant message content. The authoritative path is the current [Mem0 adapter](../../src/oamb/memory_systems/mem0/adapter.py) and [wire contract](../../src/oamb/memory_systems/mem0/wire.py).

This translation establishes how OAMB sends historical time; it does not turn the OAMB run into a reproduction of Mem0's managed backend. Mem0 may still deduplicate facts, retain several versions, or use recent messages across source-session boundaries according to its own runtime behavior.

## 6. Accuracy evidence

The final OAMB result contains this balanced breakdown:

| LongMemEval type | Correct | Total |
|---|---:|---:|
| knowledge-update | 9 | 10 |
| multi-session | 8 | 10 |
| single-session-assistant | 7 | 10 |
| single-session-preference | 9 | 10 |
| single-session-user | 10 | 10 |
| temporal-reasoning | 9 | 10 |
| **Overall** | **52** | **60** |

The eight judged-wrong question IDs are `031748ae_abs`, `2ce6a0f2`, `3e321797`, `41275add`, `75832dbd`, `8cf51dda`, `gpt4_15e38248`, and `gpt4_fa19884d`. The result map contains exactly 60 unique terminal judged records, and its retained validation evidence has no failed rules, missing required rules, or issues.

Mem0's checked [public result artifact](https://raw.githubusercontent.com/mem0ai/memory-benchmarks/main/results/platform/longmemeval_results.json) contains 500 evaluations and reports 467 passes, 33 failures, zero errors, and 93.40% at `top_200`. The current [public README](https://github.com/mem0ai/memory-benchmarks/blob/main/README.md#longmemeval) reports 472/500, or 94.40%, but the inspected update does not include replacement per-question predictions and judgments corresponding to that number.

## 7. Consequences for benchmark interpretation

### 7.1 Supported claims

1. The final OAMB Mem0 result is 52/60, or 86.67%, under the balanced common OAMB protocol.
2. The checked Mem0 managed artifact is 467/500, or 93.40%, and Mem0's current public headline is 472/500, or 94.40%.
3. OAMB is arithmetically 6.73 percentage points below the checked artifact and 7.73 points below the current headline.
4. Both are legitimate end-to-end system-accuracy measurements of their stated protocols.

### 7.2 Unsupported claims

1. The percentage-point differences do not isolate Mem0 retrieval quality because the question set, backend, retrieval breadth, answer path, judge path, and evidence contract differ.
2. The OAMB LME-60 result is not a 60-question reproduction or subset score of either 500-question managed result.
3. The public files inspected here do not independently verify the 472/500 headline at per-question level.
4. The evidence does not establish intent from the missing replacement artifact.

### 7.3 Minimum controls for an attributable comparison

An attributable experiment requires identical question membership, source bytes and timestamps, Mem0 backend and dependencies, isolated state, extraction configuration, retrieval request and result limit, answer-visible context, answer and judge prompts and models, retry behavior, and denominator. Until then, the reported percentage-point differences describe complete pipelines rather than a Mem0-only effect.

## 8. Source map

| Concern | Source |
|---|---|
| OAMB dataset, models, retrieval policy, and balanced selection | [`configs/benchmark.yml`](../../configs/benchmark.yml), [`src/oamb/workloads/longmemeval.py`](../../src/oamb/workloads/longmemeval.py) |
| OAMB Mem0 ingestion and temporal translation | [`src/oamb/memory_systems/mem0/adapter.py`](../../src/oamb/memory_systems/mem0/adapter.py), [`src/oamb/memory_systems/mem0/wire.py`](../../src/oamb/memory_systems/mem0/wire.py) |
| OAMB recorded-evidence validation | [`src/oamb/artifacts/validation/mem0_evidence.py`](../../src/oamb/artifacts/validation/mem0_evidence.py) |
| Mem0 evaluation runner | [`benchmarks/longmemeval/run.py`](https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/longmemeval/run.py) |
| Mem0 answer and judge prompts | [`benchmarks/longmemeval/prompts.py`](https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/longmemeval/prompts.py) |
| Checked managed result | [`results/platform/longmemeval_results.json`](https://github.com/mem0ai/memory-benchmarks/blob/main/results/platform/longmemeval_results.json) |
| Current managed headline | [Mem0 benchmark README](https://github.com/mem0ai/memory-benchmarks/blob/main/README.md#longmemeval) |
