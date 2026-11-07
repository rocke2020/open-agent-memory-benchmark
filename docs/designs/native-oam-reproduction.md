# Native OAM reproduction mainflow

## Overview

OAMB evaluates current OAM using the published LME-500 and BEAM-10M procedure, the same DeepSeek Flash model and Qwen3 embedding model, and explicit OAM revision/configuration records. Matching exact scores is not required; comparable quality remains a measurement, not a consequence of successful execution.

## 1. Inputs and prompts

Use the existing pinned LongMemEval downloader. The pinned BEAM-10M downloader/exporter reads the two official Parquet shards, writes the same JSON serialization used by AMB, verifies the campaign JSON hash, and installs without overwriting mismatched files. Bulk data stays ignored. It reuses the existing verified-file installation function.

The LME answer and judge template bytes already match the recorded campaign. The workflow vendors the recorded BEAM answer and nugget-judge text from AMB commit `3d1514c75f7940064945edcd9865152fffb80d6f`, with attribution. Preserve the historical JSON answer schema/system message and per-rubric scoring; do not replace the published prompts with new wrappers.

## 2. Execution

`./precheck.sh --oam CONFIG --output DIR` freezes dataset, selection, prompts, model request name, runtime configuration and budgets. The same configured generative model request name must resolve for OAM extraction, answering and judging. Its documented model identity is `deepseek-v4.1-flash`; the provider may require the recorded `deepseek-flash` gateway alias. Embedding remains `qwen3-embedding:0.6b`, 1,024 dimensions. Record supplier model metadata in raw receipts.

`./run.sh --oam DIR` reuses existing serial history ingestion, query readiness and index-plan checks. LME uses fast recall and BEAM normal recall. OAM owns extraction prompts inside its pinned image; OAMB owns benchmark prompts. Current OAM tokenizer/index changes are disclosed rather than reverted.

For each question, save the high-effort first answer and high-effort judgment atomically before any policy retry. The existing completion cache and budgets own all attempts and reuse complete valid outputs; there is no new retry executor or queue.

## 3. Reviewed policy retry and resume

Failed first-pass questions require a source/context/gold review before one max-effort answer retry. A small `retry-eligibility.json` file in the run directory records the question, unchanged context hash, completed coverage/gold review, gold status and rationale. Incorrect gold blocks retry. Unreviewed failures remain explicitly pending; do not automatically retry every failed question or choose the better score.

On `./run.sh --oam DIR --resume`, eligible failed questions reuse the exact frozen context/prompt/model and obtain one max-effort answer plus high-effort judgments. A dispatched complete retry becomes the policy result even if its score falls. An interrupted retry reuses the existing paid-call cache. Original first-pass results remain unchanged.

## 4. Validation and reporting

The result validator binds both first-pass and retry outputs to actual model receipts, efforts, exact prompts, context hash, rubric coverage and recomputed scores. The report shows first-pass and policy-final scores separately and keeps all original questions in the denominator. Pending eligibility prevents claiming a complete policy-final result.

Implementation owners are [the public workflow](../../src/oamb/oam_workflow.py), [input loading](../../src/oamb/workloads/oam_inputs.py), [the OAM adapter](../../src/oamb/memory_systems/oam.py), [prompt templates](../../src/oamb/workloads/prompt_templates/) and [download scripts](../../scripts/download/). [The focused protocol helper](../../src/oamb/workloads/oam_protocol.py) owns model request schemas and retry-review admission; the existing completion cache and runner own execution and budgets.

## 5. Acceptance

Use fail-capable offline tests for immutable export/hash rejection, published prompt bytes/rendering, valid and blocked eligibility, single retry, lower retry scores, interrupted/resumed reuse, model consistency and tampered first-pass/retry evidence. No tests make paid calls or real database writes.

Run a small real evaluation through the public commands in fresh isolated services using the same models, inspect raw usage and finish reasons, independently recompute scores, verify resume reuses accepted work, and open HTML in existing Chrome. Small or partial-history diagnostics prove execution and policy behavior, not full-dataset score parity. Preserve all failed artifacts and do not rerun the completed historical campaigns.
