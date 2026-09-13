# OpenViking LongMemEval Evaluation Comparison: OAMB and Official Self-Evaluation

## Overview

1. We thank the OpenViking maintainers for publishing their LongMemEval importer, evaluator, prompts, judge, statistics script, and run instructions. Those sources make a concrete comparison possible and directly informed OAMB's use of `find`, generated-sidecar filtering, and full visible-memory reads.
2. Accuracy is the headline comparison: the saved OAMB OpenViking LME-60 result is 51/60, or 85.00%, while the pinned official self-evaluation sources publish no completed numerator, denominator, result artifact, or score. The verified official-versus-OAMB percentage-point gap therefore cannot be calculated from the inspected evidence.
3. OAMB and the official self-evaluation share the same high-level memory path: isolate one question history, ingest its conversations as OpenViking sessions, retrieve with the raw question, remove generated sidecars, read retained memories, generate an answer, and judge it against the gold answer.
4. They differ after and around that shared core: dataset selection, user authentication, session scheduling, retrieval breadth, client reranking, answer context, answer instructions, model selection, judge instructions, error handling, score denominator, and timing definitions.
5. At the exact OpenViking `v0.4.19` release, two bundled-client compatibility mismatches prevent the published scripts from executing the intended ingestion-and-retrieval path unchanged. These are facts about that released evaluation directory, not about the OpenViking memory runtime generally.

## 1. Scope and terminology

This investigation compares OAMB's `openviking-session-rest-v1` LME-60 cell with OpenViking's official self-evaluation directory at release `v0.4.19`, commit `f3afef11637f2d7c11e4b1f36ed2f90630737cdc`. It describes observable protocol and artifact facts and does not rate either project or generalize to differently configured OpenViking deployments.

| Term | Meaning in this document |
|---|---|
| OAMB fair-comparison track | The balanced LME-60 protocol that keeps the question set, answer model and prompt, judge model and prompts, retrieval-generation policy, and `top_k=150` common across Hindsight, Mem0, and OpenViking. |
| OV official self-evaluation | The importer, evaluator, prompts, judge, statistics script, and README in OpenViking's pinned `benchmark/longmemeval/openviking` directory. |
| System accuracy | Final judged-answer accuracy for the entire retrieval, answer, and judge pipeline; it is not retrieval recall alone. |
| Direct correspondence | An official prompt rule addresses the same factual pattern visible in a retained OAMB case. It does not mean the case was rerun or that its verdict would change. |

| Evidence | Bound identity |
|---|---|
| OAMB workload | Cleaned LongMemEval S source SHA-256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`; balanced manifest hash `90b2669f7b893e59d404549f5803882bcd6640ce82520a9bf09672cc79464c80`. |
| OAMB result | Saved bundle `openviking-lme60-20260913-ov-oamb/`; `results/openviking.json` SHA-256 `3316102facf98977bb9c4d0b14e9a12b5cd5a5ba2ed15665745242cd0a5c7761`. |
| OV official source | OpenViking `v0.4.19` at commit `f3afef11637f2d7c11e4b1f36ed2f90630737cdc`. |

The [pinned official directory](https://github.com/volcengine/OpenViking/tree/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking) contains source and documentation but no retained evaluation CSV. Its [README](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/README.md) documents commands and settings without publishing a completed denominator, result hash, or score.

## 2. Shared semantic core

Both paths:

1. Treat each LongMemEval question history as a separate OpenViking user memory space.
2. Send user and assistant messages without the dataset-only `has_answer` annotation.
3. Represent each source conversation as a native OpenViking session, retain message order, attach source time, commit the session, and wait for extraction tasks before evaluation completes.
4. Keep the gold answer out of ingestion, retrieval, and answer generation.
5. Query OpenViking `find` with the raw question against the question user's memory root.
6. Remove `.abstract.md` and `.overview.md` hits and read retained memory URIs before answer construction; OAMB preserves find order, while the official path can later reorder the hydrated memories through client reranking.
7. Generate an answer from retrieved memory and judge it against the LongMemEval gold answer.

The official self-evaluation's provider-facing retrieval materialization was a useful reference for OAMB. OAMB retained the shared `find → sidecar filter → full read` sequence while using its cross-provider answer and judge path.

These shared stages do not make the results interchangeable because the concrete inputs, models, prompts, retrieval post-processing, error treatment, and score denominators differ.

## 3. Concise differences

Accuracy is the largest visible reporting difference motivating this investigation, but only the OAMB value is bound to an inspected result artifact. The confirmed comparison is 85.00% for OAMB versus no published numerical result in the pinned official sources; it is not a source-confirmed numerical gap between two scores.

| Layer | OAMB LME-60 | OV official self-evaluation |
|---|---|---|
| Accuracy | 51/60, or 85.00%, from 60 unique terminal judged records and validated result evidence. | No completed numerator, denominator, result artifact, or score is published in the pinned directory or README. |
| Question set | Frozen balanced 60: ten questions from each of six types, with source and manifest hashes. | Reads a caller-provided JSON or CSV; evaluates every row by default, with optional index or prefix-count selection. |
| Question identity | Fresh user and matching user API key for each question occurrence. | Stable question-derived user label sent through an identity header. |
| Session scheduling | Waits for each source session to become terminal before submitting the next session for that question. | Recommended deferred mode can submit several sessions before waiting for their tasks. |
| Retrieval | Generation-free `find`, `context_type="memory"`, `limit=150`, no reranker, no provider-specific context ceiling. | `find`; bare defaults are 10 retrieved files, 10 reranked files, and 4,000 context characters; the README full example uses 50, 10, and 30,000. |
| Answer input | Common evidence format and provider-neutral prompt; question type is not disclosed to the answer model. | Full memory text after optional client reranking and character filtering; prepends the dataset `question_type` and applies the official answer prompt. |
| Answer model | Common frozen OAMB answer role across Hindsight, Mem0, and OpenViking. | VLM selected by the caller's OpenViking configuration. |
| Judge | Common frozen OAMB judge role and original category-sensitive LongMemEval rubrics. | Default OV judge prompt or optional strict prompt; default judge model `doubao-seed-2-0-pro-260215`. |
| Errors | Failed or unknown stages remain non-scored and retain raw attempt evidence. | Retrieval or answer exceptions become response text; judge API or parse errors are stored as `WRONG`. |
| Denominator | Requires all 60 selected questions to be terminal and judged for the reported LME-60 result. | Statistics use `CORRECT / (CORRECT + WRONG)`; other rows are reported separately. |
| Timing | Separates ingestion, retrieval, answer, and judge stages. | `time_cost` spans client setup, retrieval, reads, reranking, and answer generation. |

The exact settings above are visible in the official [README](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/README.md#full-eval), [evaluator](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/run_eval.py), [judge](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/judge.py#L145-L186), and [statistics script](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/stat_judge_result.py#L32-L49).

## 4. End-to-end flows

The two implementations share a stage graph but place different contracts around each stage.

### 4.1 OAMB

```text
frozen balanced LME-60 manifest
  → fresh authenticated OpenViking user for one question occurrence
  → chronological native sessions, each terminal before the next
  → generation-free find(question, memory root, limit=150)
  → remove generated sidecars and read every retained memory URI
  → common provider-neutral answer model and prompt
  → common category-sensitive LongMemEval judge
  → atomic question result and independently validated capsule
```

### 4.2 OV official self-evaluation

```text
caller-provided LongMemEval file
  → stable question-derived user label
  → source-order native sessions with optionally deferred task waiting
  → find(question, memory root, limit=10 by default or 50 in README)
  → remove generated sidecars and read retained memory URIs
  → optional client reranking; keep 10 in the README example
  → 4,000-character default or 30,000-character README context budget
  → question type plus official answer prompt and configured OpenViking VLM
  → default judge prompt or optional strict prompt and configured judge model
  → CSV rows and statistics over CORRECT plus WRONG rows
```

## 5. Official answer and judge instructions

The official [answer prompt](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/longmemeval_prompts.py#L16-L134) contains explicit task-specific instructions:

1. Compute relative time from the supplied question date, apply inclusive time windows, and use specified interpretations for phrases such as “last month.”
2. Distinguish exact entities and role titles; abstain when the question names a different entity or title from the memory.
3. For counts, enumerate dated items, apply the exact verb or qualifier, scan all memories, deduplicate, and perform another full scan.
4. For suggestions, identify what the user does, avoids, and wants to explore, then check every suggestion against the avoided items.
5. Use the most recent value for updates while distinguishing genuine updates from different people or contexts.
6. Treat a provided sequence of song notes as the requested chord progression when chords are otherwise absent.

The evaluator also [prepends the dataset question type](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/run_eval.py#L189-L204) to this answer input.

The official [default judge prompt](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/longmemeval_prompts.py#L137-L205) states “When in doubt, lean toward yes” and specifies these acceptance rules:

1. Accept semantic equivalents and supersets unless added details are shown to be wrong.
2. For lists, accept synonyms, related sub-concepts, omitted optional alternatives, and one item when two listed items serve the same purpose.
3. Accept ranges containing the gold quantity, rough quantities, approximate unit conversions, and off-by-one day, week, or month values.
4. Accept notes in place of chords when justified, same-day ordering swaps, and older information when the current value is also identified.
5. Evaluate preference answers by their overall direction; minor incidental references to avoided items can remain acceptable.
6. Explicitly distinguish disinterest in general AI topics from interest in specific topics presented at general-AI conferences.
7. Accept multiple phrasings that communicate abstention when the gold answer is unavailable.

The [judge CLI](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/judge.py#L163-L186) selects this prompt by default and exposes the separate strict prompt through `--strict-prompt`.

## 6. OAMB result and direct case correspondences

The saved OAMB map contains 60 unique terminal judged records: 51 correct and 9 wrong. Its retained validation evidence has disposition `validated` with no missing or failed required rule.

| LongMemEval type | Correct | Total |
|---|---:|---:|
| knowledge-update | 9 | 10 |
| multi-session | 9 | 10 |
| single-session-assistant | 5 | 10 |
| single-session-preference | 8 | 10 |
| single-session-user | 10 | 10 |
| temporal-reasoning | 10 | 10 |
| **Overall** | **51** | **60** |

Four judged-wrong OAMB cases have a direct textual correspondence with official prompt rules:

| Case | Retained OAMB facts | Corresponding official rule |
|---|---|---|
| `031748ae_abs` | The question asks about a “Software Engineer Manager”; the memory and gold distinguish “Senior Software Engineer.” OAMB answered “4 engineers,” noted the title mismatch, and received `No`. | The answer prompt directs the model to abstain when the question uses a different role or title from the memory. |
| `eaca4986` | The answer reproduced `C D E F G A B A G F E D C` but called it a melody, said the chord progression was unavailable, and received `No`. | The answer prompt says song notes count as the chord progression when chords are absent; the default judge also permits notes in place of chords when justified. |
| `1c0ddc50` | The gold says the user wants podcast genres beyond true crime or self-improvement. OAMB suggested those two genres and received `No`. | The answer prompt requires listing avoided or disliked items and checking every suggestion against them. |
| `75832dbd` | The gold asks for healthcare AI, especially medical-image analysis, and excludes general AI topics. OAMB suggested named broad AI conferences plus unrelated multi-agent and multimodal reinforcement-learning work and received `No`. | The default judge explicitly treats general AI topics and specific topics presented at general-AI conferences as different preference categories. |

These are correspondences between the official prompt text, the pinned dataset rows, and retained OAMB answers. No retained run applies the official answer prompt or judge to the same four OAMB evidence sets, so the document does not claim that any verdict would change.

Of the other five OAMB errors, four lacked the exact required gold detail in answer-visible evidence (`3e321797`, `41275add`, `e3fc4d6e`, and `fca762bc`), while `gpt4_15e38248` contained relevant evidence but produced an answer that did not satisfy the common judge. All nine wrong records have `answer.protocol_disposition=normal_stop`; their `retrieval.visible_truncated_count` measurements have `status=measured` and `value=0`.

The nine wrong cases averaged 122.67 native candidates and 15,279.78 answer-visible tokens; the 51 correct cases averaged 124.02 candidates and 15,491.53 tokens. The artifact therefore does not show a simple candidate-count or visible-token separation between correct and wrong cases.

## 7. Exact-release behavior and retained boundaries

The following facts are independently checkable at the pinned release:

1. [`import_to_ov.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/import_to_ov.py#L263-L276) calls `commit_session(..., options={"telemetry": True})`, while the bundled [compatibility client](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/openviking_cli/client/_http_compat.py#L166-L181) accepts `telemetry=` and has no `options` parameter. That exact call raises `TypeError`.
2. [`run_eval.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/run_eval.py#L144-L186) checks list and attribute-based find-result forms. The bundled [SDK client](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/sdk/python/openviking_sdk/client.py#L1361-L1382) returns the result dictionary, so the fallback iterates dictionary keys and selects no memory contexts.
3. The evaluator sends the derived question user through `X-OpenViking-User`. Under the bundled [API-key authentication plugin](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/openviking/server/auth/plugins/api_key.py#L68-L105), the API key owner determines account and user authority and the assertion headers are removed.
4. The README says rerank limit zero disables reranking, while [the rerank function](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/run_eval.py#L254-L313) still invokes an available reranker and retains at least one item through `max(1, limit)`.
5. [Retrieval or answer exceptions](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/run_eval.py#L375-L520) become response text, judge failures are stored as `WRONG`, and the statistics script excludes other rows from its accuracy denominator.

A result produced after changing either compatibility mismatch is evidence for that changed execution. The observations above do not imply that the OpenViking server's memory extraction or retrieval engine has the same behavior.

## 8. Consequences for benchmark interpretation

### 8.1 Supported claims

1. OAMB OpenViking scored 51/60 under the common OAMB answer and judge protocol.
2. The official OpenViking self-evaluation shares OAMB's main provider-facing retrieval sequence.
3. The official path adds task-specific answer rules, question-type disclosure, client reranking and context filtering, a different answer model selection, a different judge prompt and model, and a different denominator policy.
4. Four official prompt rules correspond directly to four judged-wrong OAMB cases.
5. The pinned official directory does not publish a result artifact or score, and the exact scripts require two client-compatibility changes to execute their intended path.

### 8.2 Unsupported claims

1. The available evidence does not establish a reproducible official score because no official result artifact was found in the pinned directory.
2. It does not establish that any prompt, reranker, model, or judge difference caused a particular numerical gap.
3. It does not establish how the four mapped OAMB cases would be graded under the complete official path because no retained identity-matched regrade was inspected.
4. It does not establish that the released-script compatibility behavior also occurs inside OpenViking's memory extraction or retrieval engine.

### 8.3 Minimum controls for an attributable comparison

An attributable experiment requires the same frozen questions, fresh provider state, retrieved evidence, and model bindings, followed by one-variable answer-prompt, question-type, reranker, judge-prompt, or judge-model ablations. Every variant must retain all selected rows and report both the full selected denominator and the graded denominator.

## 9. Source map

| Concern | Source |
|---|---|
| OAMB dataset, model roles, retrieval policy, and balanced selection | [`configs/benchmark.yml`](../../configs/benchmark.yml), [`src/oamb/workloads/longmemeval.py`](../../src/oamb/workloads/longmemeval.py) |
| OAMB OpenViking identity, ingestion, retrieval, sidecar filtering, and full reads | [`src/oamb/memory_systems/openviking/session_adapter.py`](../../src/oamb/memory_systems/openviking/session_adapter.py) |
| OV run instructions and recommended settings | [official README](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/README.md) |
| OV ingestion and evaluation | [`import_to_ov.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/import_to_ov.py), [`run_eval.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/run_eval.py) |
| OV answer and judge behavior | [`longmemeval_prompts.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/longmemeval_prompts.py), [`judge.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/judge.py), [`stat_judge_result.py`](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/benchmark/longmemeval/openviking/stat_judge_result.py) |
| OV bundled client shapes and API-key authority | [SDK client](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/sdk/python/openviking_sdk/client.py), [compatibility client](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/openviking_cli/client/_http_compat.py), [API-key plugin](https://github.com/volcengine/OpenViking/blob/f3afef11637f2d7c11e4b1f36ed2f90630737cdc/openviking/server/auth/plugins/api_key.py) |
