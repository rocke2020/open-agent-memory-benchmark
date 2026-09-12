# Mem0 LongMemEval evaluation audit and OAMB temporal-parity design

## Overview

1. The available evidence does not support alleging that Mem0 fabricated its LongMemEval results. The checked public 500-question artifact is complete and recomputes to 467/500, or 93.4%, with failed questions retained and no direct gold-answer path into ingestion, retrieval, or answer generation.
2. Mem0's current 94.4% headline is not independently supported by the public repository artifacts inspected for this report. The public result file still contains the older 467/500 result, while the change to 472/500 updated the README without publishing a corresponding 500-question raw artifact. This is a reproducibility and evidence gap, not evidence of dishonesty.
3. Mem0's published score is an end-to-end system result: managed Mem0 retrieval, GPT-5 answer generation, a LongMemEval-specific reader prompt, and a more permissive custom judge. It must not be described as pure retrieval accuracy or compared directly with a provider score produced by OAMB's common answer and judge path.
4. OAMB's current Mem0 diagnostic result of 22/30 has a serious timestamp-fairness defect. OAMB preserves the same source-session timestamp for every provider, but its Mem0 adapter does not transmit that time to Mem0. Historical events are consequently extracted and stored using the benchmark execution date.
5. The selected correction changes OAMB only. It keeps Mem0 2.0.19, the original LongMemEval rows, the original two-message pairs, the common OAMB answer and judge path, and `top_k=150`. For each Mem0 pair, OAMB supplies the source timestamp as both storage metadata and a generic highest-priority extraction instruction. A fresh timestamp-sensitive case passed the semantic gate, and the subsequent isolated LME30 run completed at 27/30, or 90.0%.

## 1. Scope and terminology

This document separates three questions that otherwise produce misleading accuracy comparisons.

- **Mem0 managed evaluation** means the public `memory-benchmarks` LongMemEval path using Mem0's managed platform and its documented answer and judge prompts.
- **Mem0 OSS evaluation** means a self-hosted Mem0 release. OAMB pins Mem0 `2.0.19` at source commit `dc82354e143c2581d505d581a00286d6ef8c3605` with PostgreSQL/pgvector storage.
- **Temporal parity** means that every provider receives the same canonical source-session timestamp and uses it as the observation time for extraction and as the stored time of the resulting memory. It does not mean different provider APIs use identical JSON field names.
- **System accuracy** means the final judged answer accuracy of retrieval plus answer generation plus judging. It is not a direct measurement of retrieval recall.
- **Adapter-level temporal translation** means OAMB maps its canonical `SourceUnit.occurred_at` into fields already supported by the selected provider API without modifying provider source code or the benchmark dataset.

The inspected OAMB diagnostic and Mem0 public artifacts used the same 30 question IDs, but they did not use the same backend, temporal semantics, extraction dependencies, answer prompt, answer model, judge prompt, judge model, or retrieval limit. Their score difference is therefore diagnostic rather than an attributable provider ranking.

## 2. What Mem0's original LongMemEval evaluation measures

Mem0's public runner sorts sessions chronologically, removes dataset-only annotations, ingests consecutive user/assistant pairs, gives every question an isolated user scope, searches the question, generates an answer, and then judges that answer. The default public runner requests `top_k=200` and defaults both answerer and judge to GPT-5. The relevant source is [`benchmarks/longmemeval/run.py`](https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/longmemeval/run.py).

This is a legitimate end-to-end product evaluation, but its result includes more than memory retrieval. Mem0's [`prompts.py`](https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/longmemeval/prompts.py) instructs the answer model to rescan memory positions 30–200, broadens several time windows, and contains dataset-specific equivalences such as treating diorama work as model-kit work, potlucks and birthday parties as dinner parties, chandeliers as jewelry, and scratch grains as new layer feed. Its judge also instructs the model to avoid rejecting answers too quickly, lean positive when uncertain, allow generous temporal rounding, and accept broad semantic equivalence.

Those choices may improve the usefulness of Mem0's benchmark system, but they prevent the resulting number from being interpreted as retrieval-only accuracy. A fair OAMB provider comparison must keep one common answer and judge process across providers; a separate vendor-protocol reproduction may use Mem0's reader and judge but must be labeled separately.

## 3. Mem0 capability boundary in the inspected OAMB runtime

The selected Mem0 source is not an old pre-v3 implementation. It contains the v3 additive extraction and hybrid retrieval pipeline. The effective runtime observed for the OAMB diagnostic nevertheless exposed only part of the system behind Mem0's managed result.

| Capability | Inspected OAMB Mem0 OSS runtime | Mem0 managed evaluation boundary |
| --- | --- | --- |
| v3 ADD additive extraction | Enabled | Enabled |
| Semantic retrieval | Enabled | Enabled |
| PostgreSQL/pgvector keyword/BM25 retrieval | Enabled, with raw-text fallback because spaCy lemmatization was unavailable | Managed multi-signal retrieval |
| spaCy | Not installed in the inspected runtime | Not publicly reproducible from the managed artifact |
| Entity extraction and entity boost | Effectively disabled because entity extraction returns empty results without spaCy | Documented as part of the managed retrieval system |
| Native reranker | Disabled in OAMB | Managed implementation is not fully disclosed by the public result artifact |
| Historical observation timestamp | Not transmitted by the OAMB Mem0 adapter | Accepted by the managed v3 path |
| Managed temporal and graph processing | Not present in the selected OSS path | Available only in the managed product path described by Mem0 |

The public Mem0 OSS evaluation image installs spaCy and its English model, while its requirements reference a mutable `feat/v3-pipeline` branch rather than the exact source commit used to produce the saved result. That branch was no longer available on the canonical remote when inspected. The public recipe therefore does not currently reconstruct a pinned OSS benchmark runtime.

## 4. Honest assessment of the published accuracy evidence

The checked public platform artifact contains 500 rows with 500 unique question IDs, 467 passes, 33 failures, and no omitted error rows. Its arithmetic recomputes to 93.4%. These facts argue against simple result filtering or denominator manipulation in that artifact. The runner strips `has_answer` during ingestion, sends only the question to search, and does not provide the gold answer to answer generation; the gold answer enters only the judge, as expected for judged accuracy.

Mem0's current [evaluation documentation](https://docs.mem0.ai/core-concepts/memory-evaluation) and repository headline report 472/500, or 94.4%, for the managed top-200 evaluation. The checked [`longmemeval_results.json`](https://raw.githubusercontent.com/mem0ai/memory-benchmarks/main/results/platform/longmemeval_results.json) still reports 467/500, or 93.4%. The [94.4% update commit](https://github.com/mem0ai/memory-benchmarks/commit/4b61c5d31b9c668a12b4f5e78064248a02c82d2b) changes the README but does not add the replacement per-question predictions, judgments, retrieved evidence, runtime configuration, or hashes.

The correct conclusion is narrow: the 94.4% result may represent a genuine internal run, but the current public repository does not provide enough matching evidence to verify or independently reproduce that specific number. This report does not infer intent from that missing evidence and does not describe the result as fabricated.

## 5. The OAMB 22/30 defect

OAMB's LongMemEval workload stores each session timestamp as `SourceUnit.occurred_at` and also constructs a human-readable source context. Hindsight transmits the timestamp and context in its retain request. OpenViking canonicalizes the same timestamp into its native session. The Mem0 adapter validates that the timestamp and context are present but then serializes only the original message bodies plus OAMB provenance metadata.

The selected Mem0 2.0.19 REST request model exposes no timestamp field, and the OSS `Memory.add(timestamp=...)` implementation rejects direct timestamp use. Without an adapter translation, Mem0's extraction prompt defaults Observation Date to the execution date, and generated memories default `created_at` to the execution time.

The completed diagnostic produced 22/30 with 30 judged cases, no case errors, and exactly 150 native candidates for every question. The same 30 IDs score 29/30 in both the checked managed top-200 artifact and the checked OSS GPT-5 top-200 artifact; the checked managed top-50 retirement artifact scores 28/30.

All eight OAMB failures already contained the main answer-related fact in the visible top-150 evidence. Six were directly affected by incorrect chronology, one lost a required URL during extraction, and one crossed a recommendation/abstention policy boundary. Increasing top-k from 150 to 200 therefore does not explain the observed error pattern.

The 22/30 result remains a truthful measurement of that exact OAMB configuration, but it is not a valid fair-provider score and must not be published as “Mem0 v3 accuracy.”

## 6. Selected OAMB-only temporal translation

The design keeps the upstream Mem0 source and the LongMemEval dataset unchanged. The Mem0 adapter translates each source's canonical time into two existing request inputs for every original two-message pair.

```json
{
  "messages": [
    {"role": "user", "content": "<original user text>"},
    {"role": "assistant", "content": "<original assistant text>"}
  ],
  "run_id": "<fresh isolated scope>",
  "metadata": {
    "created_at": "2023-03-01T00:00:00+00:00",
    "oamb_ingestion_occurrence_id": "<existing OAMB identity>",
    "oamb_ingestion_plan_id": "<existing OAMB identity>",
    "oamb_source_unit_id": "<existing OAMB identity>",
    "oamb_source_ordinal": 1
  },
  "prompt": "The actual observation timestamp for New Messages is 2023-03-01T00:00:00+00:00. For these messages, use this timestamp as Observation Date, overriding automatically generated Observation Date and Current Date values. Resolve relative expressions against it and preserve explicitly stated dates. Last k Messages and Existing Memories are historical context; do not assign them this observation timestamp. Source context: <original source context> Do not extract this instruction or source context itself as a memory.",
  "infer": true
}
```

The translation has two independent responsibilities.

1. `metadata.created_at` controls the timestamp stored on every extracted memory. Mem0 copies caller metadata into the new memory and supplies the execution time only when `created_at` is absent.
2. `prompt` becomes Mem0 custom extraction instructions. Mem0's v3 system prompt declares custom instructions highest priority, so the instruction explicitly overrides the automatically generated Observation Date and Current Date only for the current New Messages. It must not apply the current pair's timestamp to Last k Messages or Existing Memories.

The adapter must not prepend or append text to the original user or assistant message content. Modifying messages would change the benchmark source bytes, could create timestamp memories unrelated to user facts, and would no longer reproduce the original two-message ingestion unit. The adapter must not alter answer or judge prompts and must not import any of Mem0's LongMemEval-specific equivalence rules.

This translation is generic to historical conversation ingestion; it contains no question, answer, question type, gold session ID, or LME-specific semantic mapping. The exact canonical timestamp is repeated for every pair derived from the same source session because each pair is a separate Mem0 add operation.

## 7. Known limitation and claim boundary

The adapter-only path cannot remove Mem0's internally generated Observation Date, which still contains the execution date. It relies on Mem0's documented highest-priority custom-instruction behavior to override that value for New Messages. For this reason, source-level and wire-level inspection alone do not establish temporal parity.

Mem0 keeps recent messages under the question-wide run scope. Those messages can cross source-session boundaries, their message-history timestamps are execution timestamps, and their formatted history does not retain each original source date. Applying the current pair's source timestamp to Last k Messages would therefore corrupt earlier-session chronology. The custom instruction is deliberately scoped to New Messages; the single-case gate must inspect a source-session boundary and disclose any unresolved pronoun or cross-session reference error.

The selected Mem0 path is additive. It may deduplicate an extracted fact or retain multiple versions rather than update one record in place. A repeated fact can therefore retain the earlier source date, and a newer statement can coexist with it. OAMB must inspect provider behavior rather than require one memory per pair or silently rewrite these outcomes.

Correct `created_at` is necessary but not sufficient. OAMB currently sends memory text, not the provider's `created_at` field, to the answer model. The extracted memory text must therefore contain the correct resolved temporal fact; a correct storage date alone cannot turn an incorrect extracted sentence into a correct answer.

Before a full LME30 dispatch, one fresh timestamp-sensitive LongMemEval case must prove all of the following:

1. Focused wire and REST black-box tests prove that every outbound pair keeps the original two message objects and includes the exact source timestamp in both `metadata.created_at` and the custom instruction.
2. The same request-contract tests prove that the generic override contains no LME question, answer, gold-session, or category hints.
3. Retrieved memories report the original source date rather than the execution date.
4. Relative temporal content resolves against the source date.
5. The question answer and judgment complete normally through OAMB's unchanged common path.
6. The independently reconstructed search and provider projection agree on `created_at`; a planted missing, changed, or mismatched date fails validation.

Only after this gate passes may the run be described as having demonstrated equivalent source-time semantics for the selected case. Even then, the final report must identify the mechanism as OAMB adapter-level temporal translation rather than native Mem0 OSS timestamp support.

## 8. TDD and verification plan

The production behavior begins with focused failing tests.

1. Exact wire test: an LME source timestamp produces `metadata.created_at` and the exact generic custom instruction while preserving the original pair bytes.
2. Negative wire test: missing, empty, malformed, or timezone-naive timestamps fail before request dispatch.
3. Pair test: every pair from one source carries the same canonical timestamp and source identity without changing pair count or role/content.
4. Non-LME test: a source without temporal fields retains the existing Mem0 request shape and receives no fabricated timestamp or prompt.
5. Response/projection test: `created_at` remains a reserved top-level Mem0 field while OAMB provenance metadata retains its existing closed schema.
6. Recorded integration test: a provider response created with historical `created_at` reconstructs and validates through the capsule path; changing the outbound timestamp or instruction fails the request-contract test, while changing the returned or projected date fails independent evidence reconstruction.
7. Synthetic semantics test: `yesterday`, an explicit absolute date, two source sessions with different dates, a repeated fact, and a later changed fact remain distinguishable without assigning a new session's timestamp to Last k Messages.

After focused and affected offline tests pass, the real gate uses question `1d4e3b97` in a fresh Mem0-only scope because its existing failure exercises chronology, repeated bike-related facts, and the boundary between retrieved history and the final recommendation. Its 45 sessions contain 474 turns and 239 two-message additions. No old Mem0 scope, result, capsule, or ingestion receipt is reused.

If the single case passes the semantic checks, run the fixed 30-question selection in a second fresh Mem0-only scope with `top_k=150`, the same DeepSeek Flash role bindings, the same Qwen3 embedding binding, the original chronological two-message pairs, and OAMB's common answer and judge prompts. Preserve every partial or failed artifact and stop new dispatch on an incorrect scope/model, missing timestamp fields, timestamp mismatch, malformed extraction, repeated authentication/quota failure, or unknown write outcome.

The LME30 result is accepted only with exactly 30 unique terminal cases, 30 valid judgments, five questions from each LongMemEval type, independently recomputed accuracy, exactly recorded native candidate counts, and a fresh capsule validation. A low score remains a completed result; it does not authorize prompt tuning, question-specific rules, method changes, or expansion to 500 questions.

## 9. Completed OAMB evidence

The real gate and LME30 run completed on 2026-09-12 without changing the selected method.

The fresh `1d4e3b97` gate preserved the chain-and-cassette fact as February 1, 2024, stored the extracted memories with the source session's `2024-02-20T19:01:00+00:00` timestamp rather than the 2026 execution date, returned the chain-and-cassette fact at rank 1 and the Garmin fact at rank 33 of 150, produced a correct common-path answer, received a `Yes` judgment, and passed all eight capsule-validation rules. Its capsule ID is `4879311ea7844b0b9c935dad790644f28e98ac2d73a573f27b69b8627bdfb4d7`.

The subsequent fresh Mem0-only LME30 run used resolved-plan hash `17279da8a80b74e0f0ddbc9b2e49a790737f2689ad7afeb9f30617c7b16220f6`, `deepseek-flash` for extraction, answer, and judge, Qwen3-Embedding 0.6B, and `top_k=150`. It finalized 30 isolated ingestion occurrences and 30 cases after 1,418 successful source-session ingestion operations. All 30 cases were judged, all returned exactly 150 native candidates, no case had an error stage, and the independently reopened capsule passed all eight validation rules. The final score is 27/30, or 90.0%. Its capsule ID is `2f51d45f071b10a3b098465bd040d5e08d6e524faa620c778acdb110c68a130c`.

| Question type | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| knowledge-update | 5 | 5 | 100% |
| multi-session | 4 | 5 | 80% |
| single-session-assistant | 3 | 5 | 60% |
| single-session-preference | 5 | 5 | 100% |
| single-session-user | 5 | 5 | 100% |
| temporal-reasoning | 5 | 5 | 100% |
| **Overall** | **27** | **30** | **90.0%** |

Compared with the timestamp-defective 22/30 diagnostic on the same questions, six old failures became correct, two remained incorrect, and one formerly correct question became incorrect, for a net gain of five. The paired result is directional evidence consistent with the timestamp correction, not a controlled causal estimate: the external model can still vary even with the same alias and zero-temperature request.

The three remaining failures expose distinct non-retrieval-limit boundaries.

1. `gpt4_15e38248`: the top-150 evidence contained the bookshelf, coffee table, kitchen-table repair, and Casper mattress. The answer mentioned all four but classified the mattress as bedding and returned three; the gold answer is four.
2. `41275add`: the top-150 evidence contained the correct Mayo Clinic video title but the extracted memory omitted the required YouTube URL, so the answer could not supply the complete gold answer.
3. `8cf51dda`: rank 2 explicitly contained “clinical and biological significance,” but the answer followed rank 1's narrower “patient outcomes and therapy response” wording and omitted biological significance; the common judge rejected the incomplete second objective.

The result strongly indicates that OAMB's timestamp omission was a major source of the low diagnostic score for this Mem0 OSS configuration. It does not establish managed-platform equivalence, isolate the effect of every changed answer, or reproduce the 94.4% headline.

## 10. Reporting language

Approved wording:

- “OAMB evaluated Mem0 OSS 2.0.19 with v3 additive extraction, PostgreSQL/pgvector semantic and keyword retrieval, native reranking disabled, and adapter-level historical timestamp translation.”
- “The OAMB result uses one common answer and judge process across providers and is not a reproduction of Mem0's managed LongMemEval headline.”
- “Mem0's public headline is an end-to-end managed-system score using its own LongMemEval reader and judge prompts.”
- “The public artifacts inspected here are insufficient to independently verify the current 94.4% figure; this is an evidence limitation, not an allegation of fabrication.”

Disallowed wording:

- “Mem0 faked its evaluation.”
- “Mem0 and OAMB used the same evaluation except for top-k.”
- “The OAMB 22/30 result measures fair Mem0 retrieval accuracy.”
- “Adding `created_at` alone fixes temporal extraction.”
- “Adapter-level translation is native Mem0 OSS timestamp support.”

## 11. Authoritative source map

| Concern | Source |
| --- | --- |
| OAMB source timestamp and context | [`src/oamb/workloads/longmemeval.py`](../../src/oamb/workloads/longmemeval.py) |
| OAMB Mem0 pair planning | [`src/oamb/memory_systems/mem0/adapter.py`](../../src/oamb/memory_systems/mem0/adapter.py) |
| OAMB Mem0 request and response schema | [`src/oamb/memory_systems/mem0/wire.py`](../../src/oamb/memory_systems/mem0/wire.py) |
| OAMB Mem0 recorded/capsule reconstruction | [`src/oamb/artifacts/validation/mem0_evidence.py`](../../src/oamb/artifacts/validation/mem0_evidence.py) |
| Shared retrieval ceiling | [`configs/benchmark.yml`](../../configs/benchmark.yml) |
| Mem0 2.0.19 source identity | [`provider-services/versions.env`](../../provider-services/versions.env) |
| Mem0 upstream evaluation runner | [`benchmarks/longmemeval/run.py`](https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/longmemeval/run.py) |
| Mem0 upstream answer and judge prompts | [`benchmarks/longmemeval/prompts.py`](https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/longmemeval/prompts.py) |
| Checked older managed result | [`results/platform/longmemeval_results.json`](https://github.com/mem0ai/memory-benchmarks/blob/main/results/platform/longmemeval_results.json) |
| Mem0 evaluation documentation | [Memory evaluation](https://docs.mem0.ai/core-concepts/memory-evaluation) |
| Original LongMemEval answer generation | [`src/generation/run_generation.py`](https://github.com/xiaowu0162/LongMemEval/blob/main/src/generation/run_generation.py) |
| Original LongMemEval judge | [`src/evaluation/evaluate_qa.py`](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py) |
