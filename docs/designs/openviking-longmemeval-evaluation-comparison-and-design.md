# OpenViking LongMemEval evaluation comparison and OAMB design

## Overview

1. The current OAMB OpenViking cell matches the selected provider-facing part of OpenViking's intended LongMemEval retrieval materialization: it filters generated sidecar hits and reads the complete visible content of every retained memory URI before answer generation.
2. OAMB provides stronger executable question isolation than the official self-evaluation's legacy authentication pattern. Every LongMemEval question receives a fresh OpenViking user, its own user-scoped memory root, and the corresponding user API key. The official script derives one user name per question but relies on a caller-supplied identity header while using the older root-key-style execution pattern; in API-key mode that header is not the authority for the data request. This comparison is about that older official evaluation path, not the current OpenViking runtime's supported user-key flow.
3. The fair OAMB design uses `POST /api/v1/search/find` with the raw question, the question-user memory root, `context_type="memory"`, `limit=150`, no retrieval-level filter, and no reranker. Under this exact no-session/no-reranker configuration, `find` and `search(mode="list")` execute the same raw-query QUICK retrieval, but OAMB keeps `find` because it explicitly freezes QUICK behavior, matches the official self-evaluation's provider-facing entrypoint, and cannot silently acquire session intent or THINKING reranking after a configuration change.
4. The exact OpenViking `v0.4.19` self-evaluation directory is a useful statement of intended behavior but is not a runnable golden implementation without corrections. Its importer calls the bundled compatibility client with an unsupported `options=` argument, and its evaluator treats the bundled SDK's dictionary `find` result as an attribute-based object, which yields no selected memories.
5. OpenViking self-evaluation permits more ingestion overlap because it recommends 16 concurrent sample submissions and defers waiting for session-commit tasks, while OAMB limits cross-question concurrency and waits for each question's sessions to become terminal in chronological order. These are source-level throughput differences, not measured speed ratios. The comparison therefore treats correctness, isolation, and evidence boundaries separately from performance.
6. The evaluation design requires a complete, independently validated denominator. Partial artifacts and source inspection are diagnostic evidence only and do not establish an LME-60 score.

## 1. Scope and terminology

This document compares OAMB's current `openviking-session-rest-v1` LongMemEval cell with the OpenViking repository's official self-evaluation flow. The official flow is treated as a legacy evaluation script whose identity and resume assumptions are weaker than the current OAMB isolation contract. The comparison is about the two evaluation designs, not a claim about every current OpenViking deployment.

| Term | Meaning in this document |
|---|---|
| OAMB fair-comparison track | The shared LME-60 protocol in this repository: the same 60 questions, answer model, answer prompt, judge model, judge prompts, retrieval-generation policy, and `top_k=150` for Hindsight, Mem0, and OpenViking. |
| OV official self-evaluation | The importer, evaluator, prompts, judge, and statistics scripts under OpenViking's own `benchmark/longmemeval/openviking` directory, evaluated as the legacy official script path. |
| Broad recall | The umbrella operation of locating relevant stored context. Both `find` and list-mode `search` are recall APIs; their names alone do not determine whether generation or reranking occurs. |
| List-mode search | `POST /api/v1/search/search` with the default `mode="list"`, which returns ranked hit buckets rather than an assembled prompt context. It remains non-generative without a `session_id`; with a session and enabled intent it can invoke query planning. |
| Retrieval mode | The internal `QUICK` or `THINKING` algorithm choice. `find` explicitly selects QUICK; list-mode `search` selects QUICK when no reranker is available and THINKING when a reranker is configured. This is separate from the HTTP `mode="list"` field. |
| Retrieval level | The indexed content tier selected by the optional `level` filter: L0 abstract, L1 overview, or L2 detail/content. Omitting `level` admits every tier and does not select QUICK or THINKING mode. |
| Question history | Every haystack session attached to one LongMemEval question. LongMemEval treats each question as a separate simulated user, even when two questions may contain similar source conversations. |
| Find hit | The metadata-level `MatchedContext` returned by OpenViking `find`, including URI, level, score, and `abstract`. It is not automatically the same byte sequence as reading the memory file. |
| Hydration | Reading the visible content of a returned memory URI through `GET /api/v1/content/read` after `find`. |
| OpenViking sidecar | A generated `.abstract.md` or `.overview.md` file used by hierarchical retrieval. It is retrieval infrastructure rather than an independent user-memory document. |
| Retrieval generation | A generative query-planning, rewriting, or session-intent operation inside retrieval. Query embedding and non-generative file reads are not retrieval generation. |

The selected OAMB design does not copy OpenViking's answer prompt, answer-model selection, custom judge prompt, judge-model default, or client-side reranker. Those components remain comparison evidence because they explain score differences, but they are not part of the common OAMB result.

## 2. Compared source baselines

The comparison uses immutable source identities. Current checkout names or README labels alone are insufficient because the OpenViking self-evaluation scripts and client API have drifted together without closing all compatibility gaps.

| Source | Inspected identity | Role in this comparison |
|---|---|---|
| OAMB | Current checkout, OpenViking `v0.4.19` adapter | Current fair-comparison implementation and selected design baseline. |
| OpenViking formal release | `v0.4.19`, commit `f3afef11637f2d7c11e4b1f36ed2f90630737cdc` | Runtime pin and authoritative released self-evaluation source. |

OAMB pins the cleaned LongMemEval S file by revision, byte count, and SHA-256, then selects a fixed balanced set of 60 questions. OpenViking self-evaluation loads the caller's JSON or CSV path without content identity validation and evaluates every row by default, with optional index or prefix-count filtering. The two paths can use the same source dataset family without using the same question set or denominator.

## 3. Shared semantic core

The intended provider interaction is substantially aligned before answer generation. Both evaluation designs:

1. Use one LongMemEval question history as one OpenViking user memory space.
2. Remove the dataset-only `has_answer` annotation and send only `user` and `assistant` message content to OpenViking.
3. Represent each source conversation as a native OpenViking session, preserve message order, attach the source-session date, commit the session, and wait for accepted memory extraction before evaluation is considered ready; OAMB can retain explicitly recorded skipped sessions after settled retry exhaustion and reports that history as partial.
4. Use the raw question as the retrieval query and target `viking://user/<question-user>/memories`.
5. Use `find`, not list-mode `search`, to freeze the direct QUICK path. List-mode `search` is also non-generative when called without a session, but its behavior becomes session-aware when a `session_id` is supplied and becomes THINKING retrieval when a usable reranker is configured.
6. Keep the gold answer out of ingestion, retrieval, and answer generation; it enters only the judge stage.

OAMB discards retrieval sidecars, reads each selected memory file, and then invokes the common OAMB answer and LongMemEval judge pipeline. OpenViking self-evaluation additionally intends to rerank those full texts, apply a character budget, and use a provider-specific answer and judge pipeline; those extra behaviors remain outside the fair track.

## 4. Concise difference and defect table

The retrieval contract requires full-memory materialization. Most v3 memory `abstract` values contain the link-stripped body rather than a conventional short summary, but they are capped at 50,000 UTF-8 bytes and can omit markdown link targets and any content beyond that cap. Treating them as identical to `content/read` is therefore incorrect even when many simple memories happen to produce similar text.

| Priority | Area | Current OAMB | OV self-evaluation intent or behavior | Classification and effect |
|---:|---|---|---|---|
| P1 | Memory hydration and sidecar admission | Excludes `.abstract.md` and `.overview.md`, then reads every retained URI with `offset=0, limit=-1`, binding candidate content to sealed request/response evidence. | Excludes `.abstract.md` and `.overview.md`, then reads each selected URI with `offset=0, limit=-1`. | The provider-facing retrieval materialization is aligned. |
| P1 | Recall API selection | Uses generation-free `find` with no session, no reranker, no `level` filter, and explicit `limit=150`. In the current configuration, list-mode `search` would reach the same raw-query QUICK retrieval. | Uses `find`, with a bare default limit of 10 and 50 in the README full example. | `find` is retained because it matches the official provider-facing entrypoint and hard-pins QUICK behavior; changing only the endpoint would add no current recall benefit and would make future reranker configuration behaviorally significant. |
| P1 | Released self-eval importer | OAMB uses the exact REST contract and has tests for session commit requests. | `v0.4.19` calls `commit_session(..., options={"telemetry": True})`, but its bundled compatibility method accepts `telemetry=`, not `options=`. | OV-self release defect. Exact-tag ingestion raises `TypeError` after messages were added and cannot be treated as a runnable oracle. |
| P1 | Released self-eval find parsing | OAMB parses the REST dictionary shape explicitly. | The bundled SDK returns a dictionary, while `_iter_search_contexts` reads `memories/resources/skills` as object attributes and then iterates dictionary keys; selection consequently yields zero contexts. | OV-self release defect. Exact-tag evaluation can answer with no retrieved memory while still producing rows. |
| P1 | Identity under API-key auth | Provisions one user per ingestion occurrence, derives that user's API key, verifies `/health`, and uses the key for every data request. | Derives one user per question but uses the legacy identity-header pattern rather than binding each question to its own user key. | OAMB is stronger for the compared official script. The current OpenViking runtime supports isolated user-key access; the official script does not implement that binding. |
| P1 | Answer and judge protocol | Uses one common answer model/prompt and one common judge model with original category-sensitive LongMemEval rubrics across providers. | Uses an extensive dataset-specific answer prompt, exposes `question_type` to the answer model, defaults to a deliberately lenient custom judge, and selects a separate default judge model. | Intentional benchmark difference. OV-self scores are not directly comparable with OAMB scores and these behaviors must not enter the fair track. |
| P2 | Dataset | Fixed balanced LME-60: ten questions from each of six types, immutable order and source hash. | All input rows by default; `--count` takes a prefix and the input is not hash-pinned. | Intentional scope difference; denominators and question difficulty differ. |
| P2 | Retrieval breadth | Sends `limit=150`; the server uses that value and has no hidden 100-result cap. | Bare defaults are 10; the README's full example uses find 50, rerank 10, and a 30,000-character answer context. | Intentional protocol difference. OAMB must keep 150 as requested. |
| P2 | Reranking | OpenViking `find` explicitly uses QUICK retrieval and disables server-side hotness/reranking; OAMB performs no later rerank. | Uses the same QUICK `find`, then separately constructs an optional client reranker over hydrated content. | Intentional fair-comparison restriction. Copying the client reranker only for OpenViking would change the measured system. |
| P2 | Session timestamps | Gives every message in one source session the same canonical UTC session timestamp. | Uses a timezone-naive source-session time plus an artificial one-second increment per message. | Input-translation difference. OAMB preserves the source's actual precision; OV-self manufactures intra-session time only to create stable ordering. |
| P2 | Session scheduling | Waits for each session commit to become terminal before the next chronological session in that question. | In recommended deferred mode, submits multiple sessions for the same question before waiting for their commit tasks. | Correctness/potential-throughput tradeoff. OAMB allows less overlap but closes chronological extraction; actual terminal indexing throughput is unmeasured. |
| P2 | Error handling | Fails the affected stage, retains raw evidence, distinguishes unknown outcomes, and does not turn an invalid answer or judge response into a normal scored result. | Converts retrieval/answer exceptions into response text, treats judge API/parse errors as wrong, and can swallow task-level exceptions while continuing. | Denominator and trust difference. A CSV row is weaker completion evidence than an OAMB validated case. |
| P2 | Resume | Reuses only identity-matched completed questions and reruns unfinished work in fresh isolated scopes. | Reuses a stable question/session success CSV without binding it to dataset bytes, runtime configuration, or model identities; `--force-ingest` adds into the same user. | Contamination risk in OV-self reruns; do not copy. |
| P3 | Context budget | Has no provider-specific item, token, or character ceiling after native retrieval. | Bare default is 4,000 characters; README full example uses 30,000 characters and skips whole files that would exceed the remaining budget. | Needs measurement after hydration. Any future ceiling should be provider-neutral, not an OpenViking-only scoring advantage. |
| P2 | Model output ceiling | Intentionally treats answer 8,192 and judge 1,024 as planning/accounting ceilings while non-probe `ModelRequest` sets `max_output_tokens=None`, so the HTTP client omits `max_tokens`. | Does not pin an answer ceiling and omits judge temperature by default. | Explicit common OAMB contract boundary, not an OpenViking-specific defect. Do not use the declared values to explain current response truncation or latency. |
| P3 | Timing metric | Measures memory-query work separately from answer and judge work. | `time_cost` spans client setup, find, all reads, rerank, and answer-model generation. | Metric-definition difference; values must not be compared as retrieval latency. |

## 5. End-to-end flow comparison

The flows have the same high-level stages but different correctness barriers and answer-visible data.

### 5.1 Current OAMB flow

```text
pinned LME-60 manifest
  → one fresh authenticated OpenViking user per question occurrence
  → reject any non-pristine memory root
  → canonical chronological source sessions
  → create one native session
  → add role/content messages with the session timestamp
  → commit and wait for the exact task to complete
  → verify archive, usage, provider projection, and source order
  → repeat the next session only after the previous session is terminal
  → find(question, target=user memory root, context_type=memory, limit=150)
  → remove generated sidecar hits and read each retained memory URI in find order
  → use each complete visible read as answer evidence
  → common OAMB answer model and prompt
  → common OAMB judge model and category-sensitive LongMemEval prompt
  → seal and independently validate the result capsule
```

### 5.2 OpenViking self-evaluation's intended flow

```text
caller-provided LongMemEval file
  → deterministic question-derived user label
  → source-order native sessions
  → add messages with artificial one-second timestamp increments
  → commit sessions, usually submitting many tasks before deferred waiting
  → find(question, target=user memory root, limit=10 by default or 50 in README)
  → discard .abstract.md and .overview.md
  → read every remaining memory URI in full
  → optionally client-rerank full content and keep 10
  → skip whole files beyond 4,000 characters by default or 30,000 in README
  → OpenViking-specific answer VLM and prompt
  → custom lenient judge by default, or optional custom strict judge
  → overwrite/update a CSV and summarize graded rows
```

### 5.3 Official script behavior on the compared legacy path

```text
import path
  → add session messages
  → fail at unsupported commit_session(options=...)
  → catch and log the per-session error

evaluation path
  → receive dictionary find result from bundled SDK
  → look for object attributes that do not exist
  → iterate dictionary keys as fallback contexts
  → reject every key because it has no .uri attribute
  → build an answer prompt with no memories
  → continue answer generation and judging
```

The intended flow is the useful provider-semantic reference. The exact released scripts are evidence that the self-evaluation directory needs compatibility corrections; they are not evidence that OAMB should reproduce those defects.

## 6. Identity and user-data isolation

OAMB's current identity design matches the basic LongMemEval requirement: each question is a separate simulated user and cannot see another question's history. The benchmark administrator is control-plane authority only; it is not the data-plane speaker or shared memory owner.

For an ingestion occurrence `A`, OAMB derives a user such as `oamb-<sha256(A)>`, provisions that exact user if absent, derives the corresponding user API key, verifies that `/health` resolves to the expected account and user, and binds the scope to `viking://user/oamb-<sha256(A)>/memories`. For a different question occurrence `B`, every value is different. A rerun also receives a new occurrence and therefore a fresh user instead of silently inheriting state from the previous run.

```text
LME question A
  → ingestion occurrence A
  → OpenViking user oamb-hash(A)
  → viking://user/oamb-hash(A)/memories

LME question B
  → ingestion occurrence B
  → OpenViking user oamb-hash(B)
  → viking://user/oamb-hash(B)/memories
```

The allocation accepts only the two sidecars that OpenViking `v0.4.19` creates for a new user, `.abstract.md` and `.overview.md`. Any other pre-existing URI rejects the scope as contaminated. All later session, task, projection, find, and read requests use the question user's API key; no `peer_id` message field or `X-OpenViking-Actor-Peer` header is used. Administrator list/create responses are sanitized before sealing so generated user keys and seeds do not enter artifacts.

The official self-evaluation has the correct conceptual isolation key but a weaker executable binding on its legacy path. It derives `lm_user_<md5(question_id)>` consistently during import and evaluation, but the calculated `agent_id` is only logged and the evaluation's `session_id` argument is unused. More importantly, its client supplies the user label as `X-OpenViking-User` without binding the question to a matching user key. In API-key mode the authenticated key, not that assertion header, determines the data identity; the requested target and authenticated identity can therefore disagree. OAMB binds each question's target, client, and user key together.

## 7. Ingestion, chronology, readiness, and throughput

OAMB prioritizes deterministic chronological memory construction. OpenViking self-evaluation permits more submission overlap. That creates a potential throughput advantage, but actual terminal indexing throughput is unmeasured, and copying deferred same-user task overlap into the fair track would change extraction semantics unless OpenViking first provides and OAMB verifies a per-user ordering guarantee.

| Dimension | OAMB | OV self-evaluation | Consequence |
|---|---|---|---|
| Session source order | Parses every date, rejects malformed weekday/date combinations, sorts by timestamp and original ordinal. | Preserves input array order without validation or sorting. | Equivalent only when the input is already ordered correctly. |
| Message time | One canonical timezone-aware timestamp for every message in a source session. | One timezone-naive base time with message index added as seconds. | Self-eval creates ordering detail not present in the source dataset. |
| Commit retention | Explicit `keep_recent_count=0`. | SDK default is also zero. | Semantically aligned. |
| Same-question session barrier | Creates, fills, commits, polls terminal task, validates archive/usage/projection, then advances to the next session. | Deferred mode submits later sessions before earlier commit tasks become terminal and waits after submission completes. | OAMB closes update order; self-eval permits task overlap, while terminal throughput remains unmeasured. |
| Cross-question concurrency | Current configuration allows three OpenViking histories to ingest concurrently. | README recommends sample and submission parallelism of 16. | The self-eval permits more overlap; its terminal wall-clock advantage has not been measured in this comparison. |
| Readiness | Binds task ID, session ID, archive URI, terminal status, token usage, accepted or explicitly skipped source partition, and final memory projection; skipped sources make the history partial. | Treats each task's completed status as success and records a CSV row; exceptions are logged per session. | OAMB has a stronger but more expensive readiness boundary without claiming every source always succeeds. |
| Resume | Completed units are identity-bound; an unfinished history is retried in a fresh question user. | Stable question/session keys are skipped from local records; force mode writes new sessions into the same stable user. | OAMB avoids duplicate/stale-memory contamination. |

The current OAMB pin uses OpenViking `v0.4.19` and the v3 memory pipeline. The official script's deferred high-parallel workflow permits more overlap than OAMB's chronological per-question barrier. OAMB also performs additional identity, terminal-task, projection, and evidence checks; the performance contribution of each difference is not part of this design comparison.

Performance work stays separate from the correctness fix. The first corrected real case remains session-serial. If indexing is still operationally too slow afterward, measure user provisioning, message upload, commit queue time, extraction, embedding, and projection separately; increase only cross-question history concurrency within observed provider and model limits. Same-question session overlap requires a demonstrated provider ordering guarantee and a test that later-session extraction cannot overtake earlier-session extraction.

## 8. Retrieval and answer-visible evidence

OAMB deliberately uses OpenViking's direct `find` API even though list-mode `search` is retrieval-equivalent under the current no-session/no-reranker configuration. The choice freezes one generation-free QUICK query and avoids a future reranker or session field silently changing the benchmark path.

### 8.1 Current OAMB OpenViking retrieval configuration

The current configuration is one explicit benchmark protocol rather than the OpenViking SDK's bare defaults. OAMB pins the latest formal server release selected at its release cutoff, OpenViking `v0.4.19`, by version, commit, and image digest; the retrieval request then overrides the SDK's default limit of 10 with the shared benchmark `top_k=150`.

| Setting | Current OAMB value | Effective behavior |
|---|---|---|
| Server | OpenViking `v0.4.19`, memory pipeline `v3`, API-key authentication | Reproducible released runtime with one authenticated question user per ingestion occurrence. |
| Endpoint | `POST /api/v1/search/find` | Direct semantic discovery without session context or intent analysis. |
| Query | Unmodified LongMemEval question | One raw typed query; no rewrite or decomposition. |
| Scope | `target_uri=viking://user/<question-user>/memories`, `context_type="memory"` | Only the authenticated question user's memory tree is eligible. |
| Breadth | `limit=150` | Uses the common OAMB retrieval breadth rather than OpenViking's API default of 10; there is no hidden 100-result cap. |
| Session and HTTP mode | No `session_id`; `mode` is not a `find` field | No session loading, intent analysis, query planning, or context assembly. |
| Reranker | No `rerank` section in OAMB's `ov.conf`; no client reranker | `find` remains QUICK and OAMB does not reorder provider results. |
| Retrieval ranking defaults | `hotness_alpha=0.0`, `score_propagation_alpha=1.0` | QUICK results use their vector scores without hotness blending or parent-score propagation. |
| Retrieval level | `level` omitted | L0 abstract, L1 overview, and L2 detail records may compete within the same provider limit. |
| Materialization | Filter `.abstract.md` and `.overview.md`, then `content/read?offset=0&limit=-1` for each retained URI | The answer receives complete visible ordinary-memory content in provider order, not the vector-index abstract. |

### 8.2 `find` versus `search(mode="list")`

Both APIs are forms of recall. HTTP `mode="list"` selects the ranked-hit response shape; it is not the internal QUICK/THINKING retrieval mode and does not itself cause generation.

| Condition | `find` | `search(mode="list")` |
|---|---|---|
| No `session_id` | Uses one raw query. | Uses one raw query and performs no intent generation. |
| No usable reranker | Explicitly selects QUICK. | Lets the retriever choose its default, which resolves to QUICK. |
| Usable reranker configured | Still explicitly selects QUICK. | Resolves to THINKING and can perform hierarchical reranking. |
| `session_id` with `retrieval.enable_intent=true` | Not supported and never enters intent analysis. | Loads bounded session context and can generate a multi-query plan. |
| Default public result | Memory/resource/skill hit buckets and total. | The same hit buckets and total when no query plan exists; it also retains an internal `QueryResult` envelope that can be exposed through provenance. |
| Other execution difference | Does not read the target abstract for planning. | May read the target abstract even when no session context ultimately triggers intent analysis. |

With the same query, scope, filter, level, limit, no session, and no usable reranker, both paths submit one raw typed query to the same QUICK vector branch. They are therefore retrieval-equivalent for the current OAMB configuration, but they are not identical APIs.

OAMB chooses `find` for three concise reasons:

1. It hard-pins QUICK behavior even if a reranker is configured later, keeping the frozen comparison from drifting.
2. It matches the provider-facing retrieval entrypoint used by OpenViking's official LongMemEval self-evaluation before that script's separate client reranking and answer pipeline.
3. List-mode `search` adds no current recall benefit without a session or reranker, while adding target-abstract work and a configuration-sensitive path to intent or THINKING retrieval.

### 8.3 Retrieval levels and visible content

The optional `level` filter is independent of both API choice and retrieval mode: L0 is a directory abstract stored as `.abstract.md`, L1 is a directory overview stored as `.overview.md`, and L2 is ordinary detail/content. OAMB deliberately omits `level`, so all tiers compete inside `limit=150`; that limit changes only result count and does not disable or select a level. Because `find` explicitly selects QUICK, those eligible level records participate in one vector search rather than an L0-to-L1-to-L2 recursive traversal. OAMB then removes only the two generated sidecar basenames and does not over-request to backfill their slots.

A find hit contains a URI, retrieval level, score, tags, and `abstract`. OpenViking reconstructs L0 and L1 hits as `.abstract.md` and `.overview.md` URIs. Its v3 memory writer stores the link-stripped memory body in the vector record's `abstract` field and truncates that field above 50,000 UTF-8 bytes. `GET /api/v1/content/read` instead returns the visible memory file: it removes the reserved `MEMORY_FIELDS` trailer while retaining normal markdown content and links. OAMB uses that visible read for retained candidates.

This means `abstract` is often richer than a short summary, but it is not byte-equivalent to the content OpenViking's self-evaluation intends to give the answer model. The differences that can affect LongMemEval include:

1. Markdown link destinations and linked identifiers removed while producing the vector abstract.
2. Text beyond the 50,000-byte abstract cap.
3. Generated L0/L1 sidecar hits that are retrieval aids rather than independent memories.
4. Formatting or visible fields present in the memory file but not in the stripped vector text.

The OAMB common evidence renderer adds deterministic `F#` labels and preserves native order, score identity, content hashes, and request evidence. That layer renders the exact candidate content it receives. The session adapter now assigns the corresponding visible `content/read` result to every retained hit.

The official self-evaluation's intended retrieval materialization is the selected provider reference for this boundary: filter the two sidecar basenames, call `read(uri, offset=0, limit=-1)` for every remaining hit, retain relative find order, and then build answer context from the returned visible strings. OAMB adopts those steps while retaining its own failure and evidence semantics. It does not copy the official script's behavior of placing `[READ ERROR]` text into the answer prompt; an unreadable selected memory makes the retrieval attempt failed or unknown according to the sealed HTTP outcome.

Hydration adds local OpenViking read calls to the memory-query stage and can increase retrieval latency and answer-context size. That increase is honest: a metadata-only vector response is not the full retrieval product used for answering. The find response remains the primary raw reference, and every ordered content-read request/response pair becomes supporting evidence so capsule validation can reconstruct candidate URI, rank, score, and exact visible content.

## 9. Answer, judge, dataset, and score comparability

OAMB and OpenViking self-evaluation measure different end-to-end systems after retrieval. Their scores must not be compared as if only the OpenViking adapter changed.

| Layer | OAMB fair-comparison track | OV self-evaluation |
|---|---|---|
| Question set | Frozen balanced 60, exactly ten from each LongMemEval type. | Every input row by default, normally the 500-question cleaned S set; optional index or prefix count. |
| Answer context | Provider candidates rendered with common deterministic OAMB evidence formatting and no provider-specific context ceiling. | Hydrated memory text, optionally client-reranked, with a 4,000-character bare default or 30,000-character README example. |
| Answer prompt | Short provider-neutral instructions covering exact speaker, counting, updates, relative time, preferences, unsupported facts, and partial evidence. | Long dataset-specific instructions and hand-authored equivalences, plus the dataset's `question_type` prepended to the model input. |
| Answer model | Common frozen OAMB answer role, model identity, temperature, thinking effort, and retries; the declared output ceiling is currently accounting-only and is not sent as `max_tokens`. | The VLM from the caller's OpenViking CLI configuration, without a benchmark-pinned identity in the CSV. |
| Judge prompt | Original category-sensitive LongMemEval templates, including separate ordinary, temporal, update, preference, and abstention rules. | Deliberately lenient custom prompt by default; optional custom strict prompt. |
| Judge model | Common frozen OAMB judge role and exact `yes`/`no` output contract. | Defaults to `doubao-seed-2-0-pro-260215`; parses either a JSON label or the final yes/no line. |
| Failure result | Failed or unjudged case remains visibly non-scored and retains attempt evidence. | Retrieval errors become answer text; judge API/parse errors become `WRONG`; ungraded rows are excluded from summary accuracy. |

Several OV-self prompt details can materially raise or otherwise change accuracy independently of memory quality. The answer model is told dataset-specific equivalences, told to scan all memories and specific deep positions, shown the question type, and required to produce hidden-style `<mem_thinking>` reasoning that remains present in the CSV response judged later. The default judge is explicitly told to lean toward yes when uncertain and accepts generous numerical, date, list, preference, and abstention equivalences. These are valid choices for a vendor end-to-end recipe, but importing them only for OpenViking would make the common provider comparison unfair.

The selected design therefore changes neither OAMB answer generation nor judging. An OAMB OpenViking score means: OpenViking `v0.4.19` memory extraction and generation-free retrieval, followed by the same answer and judge pipeline used for Hindsight and Mem0. An official self-evaluation score means the complete legacy vendor recipe and must be reported separately with its exact dataset, client, runtime, prompt, reranker, context budget, answer model, judge mode, and judge model.

## 10. Defects and special behavior in OpenViking self-evaluation

OpenViking's self-evaluation contains useful product-specific ideas and several script-level defects. OAMB should copy verified semantics, not literal implementation behavior.

1. **Importer/client signature mismatch:** The released importer correctly uses `options={"created_at": ...}` for `add_message`, because the bundled SDK accepts that shape, but incorrectly uses `options={"telemetry": True}` for `commit_session`. The bundled compatibility method exposes `telemetry` directly and has no `options` or catch-all keyword argument. The call raises before commit. This mismatch already existed before `v0.4.19`; it is not evidence of a new v0.4.19 runtime regression.
2. **Find-result shape mismatch:** The released synchronous SDK returns the `result` dictionary from `/api/v1/search/find`. The self-evaluator checks list and object-attribute forms but not a dictionary's `memories`, `resources`, and `skills` values. Its fallback iterates dictionary keys, each key lacks `.uri`, and the evaluator selects no contexts. An older object-returning client can satisfy the script, but the exact released client cannot.
3. **Identity header is not authority in API-key mode:** `user=<derived-id>` becomes `X-OpenViking-User`. The API-key authentication plugin removes and ignores that assertion header after resolving the actual key owner. A correct API-key deployment needs a real key for each question user, as OAMB now provides.
4. **Documented rerank disable is not implemented:** The README says `--single-search-rerank-limit 0` disables reranking. The code still constructs a configured reranker, performs reranking when more than one context exists, and slices with `max(1, limit)`, leaving one reranked item. Absence or failure of the reranker instead returns every input context, so the same CLI value has configuration-dependent effective breadth.
5. **Bare defaults differ from the recommended run:** The script defaults to find 10, rerank limit 10, and 4,000 context characters; the README full run uses find 50, rerank 10, and 30,000 characters. A result labeled only as “OV self-eval” is under-specified.
6. **Prompt chronology metadata is dropped:** The formatter can group memories by `created_at` and show their URIs, but the conversion immediately before it passes only memory text, score, and raw rank. The answer prompt consequently receives neither `created_at` nor URI from this path even though its prose says memories are sorted newest-first. Actual find order is relevance order.
7. **Question type enters answer generation:** The evaluator prepends `Question Type: <dataset label>` to the answer prompt. This is benchmark-only supervision unavailable in normal product use and absent from OAMB's common prompt.
8. **Errors can look like evaluated answers:** A retrieval or answer exception becomes `[SINGLE SEARCH ERROR] ...` response text and the row is still written. Judge API and parse errors return false and are stored as `WRONG`. The statistics script then calculates accuracy over `CORRECT + WRONG`, while other ungraded rows are excluded.
9. **Deferred submission weakens chronological proof:** Session commit requests for one user can be accepted before earlier sessions finish extracting. The script later waits for all tasks, but it does not verify that the provider applied same-user updates in chronological terminal order.
10. **Resume identity is local and mutable:** Import success keys contain question and session identity but not dataset bytes, runtime version, extraction configuration, embedding model, or VLM identity. `--force-ingest` adds new native sessions into the same stable question user rather than creating a fresh scope.

The compared `v0.4.19` official script has importer and find-result shape mismatches. These findings limit that script as reproducibility evidence; they do not imply that OpenViking's memory runtime has the same defects.

## 11. Selected OAMB OpenViking evaluation design

The selected change is narrow: correct OpenViking retrieval materialization while preserving the common comparison contract. It does not change the dataset, `top_k`, extraction configuration, answer model, answer prompt, judge model, judge prompt, or result denominator.

### 11.1 Frozen invariants

The corrected OpenViking cell must preserve all of these invariants:

1. One fresh authenticated OpenViking user per LongMemEval question ingestion occurrence.
2. One native OpenViking session per source session, in canonical chronological order, with the original user/assistant role and content bytes and the canonical source-session timestamp.
3. Terminal completion of one session's commit task before the next source session in that question begins.
4. Raw question bytes as the single `find` query.
5. `target_uri` equal to the question user's memory root, `context_type="memory"`, `limit=150`, no `session_id`, no query rewrite, no intent search, and no reranker.
6. Provider-returned relative order preserved after removal of generated sidecars; no OAMB relevance sorting or score transformation.
7. Common OAMB visible-evidence formatting, answer model/prompt, judge model/prompts, retries, and reporting for every provider.
8. Fail-closed identity, response-shape, read, and evidence validation; no secret-bearing user credentials in artifacts.

### 11.2 Corrected retrieval flow

```text
raw LongMemEval question
  → POST /api/v1/search/find
       target_uri = viking://user/<question-user>/memories
       context_type = memory
       limit = 150
  → validate exact response shape, user-root containment, finite scores, and unique URI/level identities
  → remove only hits whose basename is .abstract.md or .overview.md
  → for each remaining hit in provider order
       GET /api/v1/content/read?uri=<hit-uri>&offset=0&limit=-1
       use normal visible read, not raw=true
       require a successful string result
  → number retained candidates contiguously while preserving their relative find order
  → seal find request/response plus every ordered read request/response pair
  → build common OAMB visible evidence from the hydrated strings
  → common answer model and OAMB answer prompt
  → common judge model and original type-sensitive LongMemEval judge prompt
```

Sidecar removal happens after OpenViking applies the requested limit, matching the self-evaluation's intended semantics; OAMB does not over-request to backfill removed sidecars. The sealed find response retains original positions, while `NativeEvidenceCandidate.native_rank_1_indexed` describes the contiguous answer-visible candidates. Candidate identity remains URI plus retrieval level, and the content hash must bind to the corresponding sealed read response.

The read call uses the same question-user client and authority as `find`. A returned URI outside the exact memory root, a sidecar masquerading through a malformed path, a duplicate URI/level identity, a non-string read result, or a failed/unknown read terminates the retrieval attempt. OAMB does not fall back to the abstract and does not place an exception string into answer evidence because either behavior would make successful and degraded questions indistinguishable.

`native_truncated=False` is defensible for the complete `offset=0, limit=-1` visible read. The recorded capsule validator reconstructs the find-to-read mapping and rejects a missing read, reordered read, substituted URI, changed content, sidecar candidate, cross-user URI, or content hash that does not match its sealed response.

### 11.3 What the fair track deliberately does not copy

The fair track does not copy these OV-self behaviors:

1. The OpenViking-specific answer prompt, dataset-specific equivalence rules, `question_type` disclosure, or `<mem_thinking>` requirement.
2. The custom lenient or custom strict OV-self judge prompt and default Doubao judge model.
3. Server-side or client-side reranking enabled only for OpenViking.
4. OpenViking-specific 4,000/30,000-character context limits.
5. Stable question users reused across runs, trusted identity headers, force-ingestion into existing state, or local CSV-only resume.
6. Deferred overlap of same-question session extraction before chronological ordering is proven.
7. Error strings as answer evidence or automatic conversion of judge infrastructure errors into wrong answers.

If a vendor-recipe reproduction is later required, it must be a separately named, separately reported experiment with its exact corrected self-evaluation code, authentication mode, full dataset, find/rerank/context settings, VLM, prompt, judge mode, and judge model. It cannot replace or be ranked inside the common OAMB three-provider result.

### 11.4 Context-size and common-model boundaries

Hydrating up to 150 memory files may make OpenViking's answer context materially larger than a metadata-only context. The first real case must record native candidate count, hydrated bytes, visible-evidence tokens, final prompt tokens, and model disposition. If the common answer model cannot accept the result, OAMB must not add an OpenViking-only truncation rule. A provider-neutral visible-evidence budget would need a separate design applied consistently to Hindsight, Mem0, and OpenViking.

The declared LongMemEval answer and judge output budgets are intentionally not transmitted in non-probe model requests under the current OAMB contract. Changing that policy would require a separate common-harness design applied to every provider cell; it is not part of the OpenViking hydration change. Reports must describe 8,192 and 1,024 as planning/accounting ceilings rather than effective HTTP `max_tokens` values.

## 12. TDD and verification plan

The correction uses fail-capable offline tests, then proves the smallest real user path before the explicitly requested LME30 paid evaluation.

### 12.1 Offline RED tests

1. A level-2 find hit has an abstract that omits a unique markdown link target while `content/read` contains it; current behavior must fail because the candidate and answer-visible evidence lack the target.
2. Find returns `.abstract.md`, `.overview.md`, and two ordinary memory URIs; current behavior must fail because sidecars are admitted and no reads occur.
3. A find response returns an ordinary memory URI outside the authenticated question root; the adapter must reject it before any read.
4. Two ordinary hits return read responses in a different completion order; the final candidates must still follow find order.
5. One content read returns a known HTTP failure and another has an unknown timeout outcome; neither path may fall back to the abstract or emit a successful evidence batch.
6. A planted recorded capsule removes, duplicates, reorders, or substitutes one content-read reference; independent validation must reject each mutation.
7. The exact retrieval request must continue to contain `limit=150`, `context_type="memory"`, the question-user root, and no session, intent, rewrite, or rerank field.
8. The answer and judge prompt hashes and model-role bindings must remain unchanged by the OpenViking adapter correction.

### 12.2 Offline GREEN criteria

The focused adapter and capsule tests pass only when every retained ordinary hit has exactly one successful visible read, every sidecar is absent from native candidates, all read request/response pairs are retained as supporting evidence, candidate order is deterministic, and the common prompt/model path remains unchanged by the provider adapter.

The affected offline suite must include the OpenViking session adapter, REST transport, retrieval request, recorded native capsule, native validation, LongMemEval workload, result projection, Ruff, formatting for changed files, mypy, public-boundary scan, and package build. Tests use recorded or fake transports and must not write to a real OpenViking service.

### 12.3 One-case real gate

After explicit authorization for provider writes and model cost, run one timestamp-sensitive or link-sensitive LongMemEval question in a fresh OpenViking user. The case must complete `preflight → user provisioning → chronological ingestion → terminal task readiness → find 150 → sidecar filtering → full reads → common answer → common judge → capsule validation → inspectable result`.

The real gate must prove:

1. `/health` resolves every data request to the exact question user.
2. The pre-ingestion root contains only the two expected preset sidecars and no other question's memory appears after ingestion.
3. Every source session reaches a terminal successful task in chronological order with no skipped source.
4. The outbound find request contains the exact shared `top_k=150` contract and the server returns no cross-user URI.
5. Every retained ordinary URI has a sealed successful visible read, and the answer evidence contains those exact bytes rather than the vector abstract.
6. The answer and judge use the same configured models and prompts as the other provider cells.
7. The final capsule validates independently and records candidate, byte, token, latency, usage, and cost evidence without exposing credentials.

This gate is diagnostic, not a score claim. Prefer a case where the stored full memory and vector abstract differ in a gold-relevant link or tail fact, but select and freeze the case before observing the answer so the test cannot silently become prompt tuning.

### 12.4 Broader acceptance

After the one-case gate passes, run the balanced LME30 OpenViking cell in fresh users. Its result requires exactly 30 unique terminal questions, five from each type, 30 valid judgments, no cross-question user reuse, no unproven skipped ingestion, a fresh validated capsule, and independently recomputed accuracy. If accuracy is below 0.80, stop further evaluation dispatch and perform a case-level diagnosis before proposing another change; do not modify OpenViking source or add an OpenViking-specific answer or judge path.

A result may remain low. Acceptance is based on executing the intended provider and common benchmark semantics, not reaching a preferred accuracy threshold. Compare cases by retrieved URI, level, abstract/full-content difference, answer-visible bytes, answer, and judge decision; do not attribute every score change to one source-level finding.

### 12.5 Performance verification

Record wall time separately for user provisioning, message upload, commit acceptance, terminal extraction wait, readiness projection, find, content reads, answer, and judge. OpenViking `v0.4.19` includes a reduced phase-1 filesystem/task-store path, internal-stat savings, request-local query-embedding caching, and phase-2 extraction batching, but its ordinary-memory updater still defaults to eight operations or a ten-second wait per account/user-scoped updater. Append-only operations may bypass that wait.

Because the updater key includes account and user, concurrent different-question users do not combine their small merge batches. OV-self deferred same-user session submissions can fill a batch that OAMB's session-terminal barrier cannot. The performance report must therefore distinguish upstream runtime improvements from harness concurrency and must not claim that the release upgrade removed OAMB's observed indexing delay without live stage timing.

## 13. Claim and reporting boundaries

Approved wording after the implementation and verification gates pass:

- “OAMB evaluates OpenViking `v0.4.19` with one fresh authenticated user per LongMemEval question, chronological native session extraction, generation-free `find` at `top_k=150`, full visible memory hydration, and the common OAMB answer and LongMemEval judge path.”
- “The official self-evaluation informed the provider-facing find, sidecar-filter, and full-read design; its answer prompt, reranker, and judge are not used in the fair OAMB score.”
- “The exact released self-evaluation scripts contain importer and find-result compatibility defects and require correction before they can reproduce their intended pipeline.”
- “OpenViking `v0.4.19` contains specific indexing-path optimizations, but current OAMB indexing speed has not been remeasured and its per-user merge wait can still dominate.”

Disallowed wording:

- “OAMB and OV self-evaluation differ only in top-k.”
- “OAMB's `top_k=150` is ignored or capped at 100.”
- “OAMB sends only tiny summaries while OV sends all memory.”
- “The missing full read proves the entire 59.52% result is wrong.”
- “OV's released self-evaluation is the runnable golden implementation.”
- “The latest OpenViking release fixed OAMB indexing speed.”
- “An OV-self score using its custom prompt and lenient judge is directly comparable to the common OAMB score.”
- “The declared answer 8,192 and judge 1,024 budgets are effective API output limits in the current runtime.”

## 14. Authoritative source map

The comparison is source-traced. Repository paths below are the implementation entrypoints; line numbers can move, so behavioral claims must be rechecked after changes.

### 14.1 OAMB sources

| Concern | Authoritative source |
|---|---|
| Dataset identity, balanced LME-60, model roles, `top_k=150`, reranking and retrieval-generation policy, concurrency, retries, and timeouts | [`configs/benchmark.yml`](../../configs/benchmark.yml) |
| LongMemEval schema, chronology, source messages, source timestamps, common answer prompt, original judge prompts, and score parsing | [`src/oamb/workloads/longmemeval.py`](../../src/oamb/workloads/longmemeval.py) |
| Per-question user provisioning, API-key binding, pristine-root validation, session ingestion/readiness, sidecar filtering, full visible reads, and task polling | [`src/oamb/memory_systems/openviking/session_adapter.py`](../../src/oamb/memory_systems/openviking/session_adapter.py) |
| Existing OpenViking L2 hydration and supporting-reference pattern for resource content | [`src/oamb/memory_systems/openviking/adapter.py`](../../src/oamb/memory_systems/openviking/adapter.py) |
| Common provider-order evidence formatting and current no-ceiling policy | [`src/oamb/workloads/visible_evidence.py`](../../src/oamb/workloads/visible_evidence.py) |
| Retrieval timing, answer/judge dispatch, result sealing, and current untransmitted model output ceilings | [`src/oamb/runtime/native_run.py`](../../src/oamb/runtime/native_run.py) |
| HTTP model payload construction | [`src/oamb/model_clients/openai_compatible.py`](../../src/oamb/model_clients/openai_compatible.py) |
| OpenViking recorded-evidence validation | [`src/oamb/artifacts/validation/openviking_session_evidence.py`](../../src/oamb/artifacts/validation/openviking_session_evidence.py) |
| OpenViking `v0.4.19`, v3 memory, VLM, embedding, auth, and absent-reranker configuration | [`provider-services/openviking/ov.conf`](../../provider-services/openviking/ov.conf), [`provider-services/versions.env`](../../provider-services/versions.env) |

### 14.2 OpenViking `v0.4.19` sources

| Concern | Pinned source |
|---|---|
| Intended LongMemEval workflow and recommended find/rerank/context settings | [`benchmark/longmemeval/openviking/README.md`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/README.md) |
| Question user identity, source sessions, message timestamps, deferred commit submission, task waiting, and importer signature defect | [`benchmark/longmemeval/openviking/import_to_ov.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/import_to_ov.py) |
| Find parsing, sidecar filter, full reads, reranking, context budget, answer VLM, concurrency, and error handling | [`benchmark/longmemeval/openviking/run_eval.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/run_eval.py) |
| OpenViking-specific answer, lenient judge, strict judge, and date/URI formatting paths | [`benchmark/longmemeval/openviking/longmemeval_prompts.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/longmemeval_prompts.py) |
| Judge client/model defaults, parsing, failures, concurrency, and resumable CSV update | [`benchmark/longmemeval/openviking/judge.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/judge.py) |
| Accuracy denominator and token/time summaries | [`benchmark/longmemeval/openviking/stat_judge_result.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/stat_judge_result.py) |
| Released SDK dictionary `find` result and content-read API | [`sdk/python/openviking_sdk/client.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/sdk/python/openviking_sdk/client.py) |
| Released compatibility-client commit signature | [`openviking_cli/client/_http_compat.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/openviking_cli/client/_http_compat.py) |
| API-key identity resolution and ignored assertion headers | [`openviking/server/auth/plugins/api_key.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/openviking/server/auth/plugins/api_key.py) |
| `find` and list-mode `search` request defaults, session-intent gate, result serialization, and routing | [`openviking/server/routers/search.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/openviking/server/routers/search.py), [`openviking/service/search_service.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/openviking/service/search_service.py), [`openviking/storage/viking_fs/_semantic.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/openviking/storage/viking_fs/_semantic.py), [`openviking_cli/retrieve/types.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/openviking_cli/retrieve/types.py) |
| QUICK/THINKING selection, effective limit, URI/level conversion, score order, and final slicing | [`openviking/retrieve/hierarchical_retriever.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/openviking/retrieve/hierarchical_retriever.py), [`openviking_cli/utils/config/retrieval_config.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/openviking_cli/utils/config/retrieval_config.py) |
| Memory vector abstract construction, link stripping, and 50,000-byte cap | [`openviking/session/memory/memory_updater.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/openviking/session/memory/memory_updater.py) |
| User-scoped merge batching, append-only fast path, and default eight-operation/ten-second window | [`openviking/session/memory/streaming_memory_updater.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/openviking/session/memory/streaming_memory_updater.py), [`openviking/session/memory/utils/streaming_batcher.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/openviking/session/memory/utils/streaming_batcher.py) |
| Release-contained phase-1, internal-stat, query-embedding, and extraction-batching changes | [`02e31f2d6`](https://github.com/volcengine/OpenViking/commit/02e31f2d6), [`2251aa136`](https://github.com/volcengine/OpenViking/commit/2251aa136), [`2cdf64c7a`](https://github.com/volcengine/OpenViking/commit/2cdf64c7a), [`acdbd7041`](https://github.com/volcengine/OpenViking/commit/acdbd7041) |
