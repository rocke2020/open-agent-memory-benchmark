# Native OAM reproduction mainflow

## Overview

OAMB evaluates current OAM using the published LME-500 and BEAM-10M procedure, the same DeepSeek Flash model and Qwen3 embedding model, and explicit OAM revision/configuration records. Matching exact scores is not required; comparable quality remains a measurement, not a consequence of successful execution.

## 1. Inputs and prompts

Use the existing pinned LongMemEval downloader. The pinned BEAM-10M downloader/exporter reads the two official Parquet shards, writes the same JSON serialization used by AMB, verifies the campaign JSON hash, and installs without overwriting mismatched files. Bulk data stays ignored. It reuses the existing verified-file installation function.

The LME answer and judge template bytes already match the recorded campaign. The workflow vendors the recorded BEAM answer and nugget-judge text from AMB commit `3d1514c75f7940064945edcd9865152fffb80d6f`, with attribution. Preserve the historical JSON answer schema/system message and per-rubric scoring; do not replace the published prompts with new wrappers.

## 2. Execution

`./precheck.sh --oam CONFIG --output DIR` records dataset, selection, prompts, model request name and runtime configuration. It derives the input fingerprint automatically; YAML does not require a manual checksum. Environment variable names and the container configuration path follow the documented service setup without per-run alias settings. Published profiles record DeepSeek Flash and Qwen3 embedding defaults. Model names are opaque user-supplied request strings: OAMB does not whitelist, normalize or compare them. Record actual request names and supplier metadata in raw receipts; the operator chooses models and assesses comparability. Embedding still must return finite vectors of the configured dimension.

`./run.sh --oam DIR` ingests each history, waits for query readiness, retrieves, answers and judges. LME uses fast recall and BEAM normal recall. BEAM provisions the required owner index through OAM's operator command; evaluation does not inspect database query plans. OAM owns extraction prompts inside its pinned image; OAMB owns benchmark prompts. Current OAM tokenizer/index changes are disclosed rather than reverted.

Published profiles retain high-effort first answers and judges. Native YAML also supports explicit `answer_model` and `judge_model` overrides, independently of OAM extraction, and `none`, `low`, `high`, `max` or `adaptive` reasoning per scoring role. Selection is explicit role YAML, then common YAML `model`, then the existing deep/light environment value. Pass the selected string unchanged; never infer aliases or substitute a matching environment name. Embedding uses YAML `embedding_model` when provided, otherwise its existing environment value. Save resolved roles as provenance without enforcing name agreement.

The shared [model parameter adapter](../../src/oamb/workloads/model_parameters.py), migrated from the private evaluator, maps `none` to disabled thinking and `adaptive` to adaptive thinking, omitting `reasoning_effort` in both cases. Explicit levels send that field; BEAM retains enabled thinking unless overridden by the selected mode. MiniMax requests retain the legacy `reasoning_split` and adaptive-thinking mapping. Only visible final content enters scoring. Native adaptive thinking and the reviewed high-to-max retry policy are separate behaviors.

Save the configured first answer and judgment atomically before any policy retry. Reuse complete valid model responses and retain failed or interrupted attempts. The operator owns evaluation cost; native OAM adds no run-wide call/token ceilings, token reservations, per-source spending override or budget records. Use the provider's default response allowance and retain the operation timeout; native model requests add no output-token cap. Reject truncated or malformed responses rather than scoring incomplete output.

## 3. Reviewed policy retry and resume

Failed first-pass questions require a source/context/gold review before one max-effort answer retry. A small `retry-eligibility.json` file in the run directory records the question, unchanged context hash, completed coverage/gold review, gold status and rationale. Incorrect gold blocks retry. Unreviewed failures remain explicitly pending; do not automatically retry every failed question or choose the better score.

On `./run.sh --oam DIR --resume`, eligible failed questions reuse the exact frozen context/prompt/model and obtain one max-effort answer plus judgments at the configured judge effort (high in published profiles). A dispatched complete retry becomes the policy result even if its score falls. An interrupted retry reuses the existing paid-call cache. Original first-pass results remain unchanged.

Resume verifies dataset bytes, OAM scope/configuration/generation, prompt identity and non-model request settings. It does not hash implementation source files or require unchanged evaluator code. Embedding resume verifies one finite vector of the recorded dimension.

Model names, endpoints and credentials do not gate resume or accepted-call reuse. Compare the existing request payload without its `model` field, retaining all other fields and stage; preserve original request and receipt bytes. Completed question results remain reusable. Unknown-outcome calls remain blocked by that same non-model request identity. Changing a model is the operator's decision, not an alias-validation problem. This follows AMB's direct [model selection](https://github.com/vectorize-io/agent-memory-benchmark/blob/28b644bd4dd653c0207c7507280439390c829155/src/memory_bench/llm/__init__.py#L21) and [request dispatch](https://github.com/vectorize-io/agent-memory-benchmark/blob/28b644bd4dd653c0207c7507280439390c829155/src/memory_bench/llm/openai.py#L31).

## 4. Validation and reporting

The result validator binds both first-pass and retry outputs to actual model receipts, efforts, exact prompts, context hash, rubric coverage and recomputed scores. The report shows first-pass and policy-final scores separately and keeps all original questions in the denominator. Pending eligibility prevents claiming a complete policy-final result.

Offline reporting exports a review worksheet and complete source views from the frozen native plan and validated results using `./run.sh --oam DIR --export-review PATH`. After source/span review, `--review-report PATH` validates the worksheet against the same saved inputs and emits a companion noise report without provider calls or changes to scored results. Reuse the private evaluator's whole-context token accounting, UTF-8 span checks, source quotations and token-weighted aggregation. Missing, uncertain and partial review remain unavailable or explicit lower bounds; annotations never imply relevance. These tools belong in OAMB; historical experiments and product-specific probes remain archived separately.

Implementation owners are [the public workflow](../../src/oamb/oam_workflow.py), [input loading](../../src/oamb/workloads/oam_inputs.py), [the OAM adapter](../../src/oamb/memory_systems/oam.py), [prompt templates](../../src/oamb/workloads/prompt_templates/) and [download scripts](../../scripts/download/). [The focused protocol helper](../../src/oamb/workloads/oam_protocol.py) owns model request schemas and retry-review admission; the existing completion cache and runner own execution and resume.

## 5. Acceptance

Use fail-capable offline tests for immutable export/hash rejection, published prompt bytes/rendering, valid and blocked eligibility, single retry, lower retry scores, interrupted/resumed reuse, opaque model-name pass-through and tampered first-pass/retry evidence. No tests make paid calls or real database writes.

Cover all five reasoning modes for LME/BEAM, MiniMax request mapping, answer-only/judge-only overrides and unchanged model-name pass-through and unrestricted names across dispatch, cache, validation and resume. Exercise offline export/report with HTTP forbidden and scored-artifact hashes unchanged. Partial or uncertain review must not publish an exact whole-run noise ratio; exact aggregation names its fully reviewed subset.

Run a small real evaluation through the public commands in fresh isolated services using the same models, inspect raw usage and finish reasons, independently recompute scores, verify resume reuses accepted work, and open HTML in existing Chrome. Small or partial-history diagnostics prove execution and policy behavior, not full-dataset score parity. Preserve all failed artifacts and do not rerun the completed historical campaigns.
