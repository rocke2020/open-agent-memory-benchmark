# Native OAM evaluation workflows

## Overview

OAMB is OAM's standalone evaluation repository. Its native workflows run chronological interaction ingestion, operation polling, non-generative retrieval, answering and judging through the public `./precheck.sh` → `./run.sh` interface. They require Docker, Compose, Python 3.11+, `uv`, an OAM image and configured model endpoints. No product checkout, private runner or provider SDK is imported during evaluation. A separate PostgreSQL volume, API and worker belong to each deployment.

| Profile | Recall | Vector index | Question isolation | Scoring |
| --- | --- | --- | --- | --- |
| LME-500 | Fast | Shared `small` | One owner per question history | Existing OAMB LME category-specific yes/no judge |
| BEAM-10M | Normal | Dedicated `large` | One owner per conversation, reused by its questions | Mean of independent 0, 0.5 or 1 rubric-item scores |

Both profiles disable Insights, Profile and the planner and use low extraction reasoning. The full profiles are [LME-500](../configs/oam-lme500.yml) and [BEAM-10M](../configs/oam-beam10m.yml). The small [LME diagnostic](../configs/oam-lme500-smoke.yml) and [BEAM diagnostic](../configs/oam-beam10m-smoke.yml) use original synthetic fixtures; their results establish execution, not benchmark accuracy.

The reproduction target is the published evaluation procedure with the same models and scores close to the published results, allowing model variability and disclosed OAM version changes. The workflow includes the recorded LME templates and BEAM answer/nugget prompts, BEAM's structured JSON answer format and document-context metadata, and the reviewed max-effort retry procedure. It preserves conversation turns, roles, source dates, original questions and provider-returned context. Current OAM owns its extraction prompts, tokenizer and index behavior; record the evaluated revision and configuration. Keep new measurements separate from the saved [LME](../eval_results/lme500/README.md) and [BEAM](../eval_results/beam10m/README.md) campaigns. See [prompt provenance](../prompt-packs/README.md) and [the mainflow](designs/native-oam-reproduction.md).

OAMB owns the reusable evaluation implementation formerly maintained with OAM's design documents: model request adaptation, LME/BEAM input and prompt handling, scoring, reviewed retries, resume, reporting and offline source/span review. Historical reports, one-off experiments and product-internal diagnostic probes remain separate; no private evaluator checkout is required.

## 1. Prepare the image and model environment

Install the locked dependencies and build the development image from committed OAM source:

```bash
uv sync --locked --all-groups
./provider-services/oam/build-image.sh /path/to/open-agent-memory 32f97cca5c5c1fd2546ed0364d2279063afc81b6
cp .env.oam.example .env.oam
chmod 600 .env.oam
```

Set `LLM_BASE_URL` and `LLM_API_KEY` in `.env.oam`. For the published evaluation's model setup, set both `LLM_LIGHT_MODEL` and `LLM_DEEP_MODEL` to the provider's identifier for **`deepseek-v4.1-flash`**. The native workflow uses OpenAI-compatible Chat Completions. The light binding supplies OAM extraction and the default judge; the deep binding supplies the default answer. Published YAML may specify a common scoring model or explicit per-role model. These are operator-selected strings: OAMB passes them unchanged and does not validate names or infer equivalence.

Set `OAMB_EMBEDDING_BASE_URL`, optional `OAMB_EMBEDDING_API_KEY`, and **`OAMB_EMBEDDING_MODEL=qwen3-embedding:0.6b`** with 1,024-dimensional vectors. Using another embedding model or dimension is a separate evaluation configuration; adjust the runtime override and benchmark's `embedding_dimensions` together and disclose the change.

Use model endpoints reachable from both the host and the containers. The preparation helper maps loopback endpoints to `host.docker.internal`, as verified on Docker Desktop. On Linux, provide a container-reachable endpoint or configure the host-gateway mapping. Linux host-local model routing has not been live-verified here. Endpoints and keys remain together in private environment files; never commit those files.

The image pins OAM source and its lockfile. Its base image and OS packages are not fully content-pinned, so the build does not promise identical image bytes. Current ingestion uses OAM's configured tokenizer; match the embedding model's own tokenizer and input/request limits before a large run. Changing these settings does not reproduce historical chunking.

## 2. Start an isolated service and run a small sample

Use a fresh output directory and Compose project name. The setup helper rejects existing project containers or its retained PostgreSQL data volume before any database mutation, creates separate non-superuser content/API/queue login roles, migrates the new database, bootstraps credentials and writes owner-only environment files. It selects a fixed free loopback API port and waits up to 60 seconds for API readiness before publishing the client environment. There are no deletion or cleanup commands.

```bash
uv run --locked python provider-services/oam/start.py \
  --dataset lme500 --env-file .env.oam \
  --image oamb/oam:0.0.1-32f97cca5c5c \
  --project oamb-native-lme-example \
  --output outputs/oam-services/lme-example

./precheck.sh --oam configs/oam-lme500-smoke.yml \
  --env-file outputs/oam-services/lme-example/client.env \
  --output outputs/oam-lme-example
./run.sh --oam outputs/oam-lme-example
```

For BEAM, start a second fresh deployment with `--dataset beam10m`, a new project/output name, and use `configs/oam-beam10m-smoke.yml` in precheck. Do not switch a running deployment's dataset override. The generated client file includes the actual bootstrap tenant; tenant names are not assumed to be `default`.

Precheck verifies the live API contract, resolved in-container configuration hash and profile, and one finite embedding vector of the configured dimension. It records the input fingerprint automatically with selected questions, model bindings, runtime identities and prompt identity in `plan.json`; no manual YAML checksum is required. It performs no generation calls. Run repeats runtime and embedding checks before writing an isolated user collection. The dataset and credentials must remain available at their recorded paths.

Run preserves source order, roles and timestamps and splits requests only to satisfy the API's message/UTF-8 byte limits. Every batch occurrence has its own stable idempotency key. It requires successful source operations with complete extraction, source/memory readiness and semantic/temporal link readiness; partial extraction fails even when the operation reports success. Failed and ambiguous artifacts remain available.

After indexing, BEAM provisions each conversation owner's dedicated index through OAM's operator command and waits for it to be ready. Evaluation then proceeds directly to retrieval, answering and judging.

## 3. Full datasets and resume

To start a full evaluation, supply the inputs and choose the corresponding full configuration:

1. Download LME using `./scripts/download/longmemeval.sh`; its full profile uses the pinned 500-question source.
2. Download/export BEAM-10M using `./scripts/download/beam10m.sh`. It reads the two official Parquet shards at revision `9b2096193fe74e2837e4713e483351e19817773c`, preserves row/shard order and nested fields, and writes `datasets/beam/10m.json` using the campaign's JSON serialization. The resulting SHA-256 must be `c16643a7cc7d93c1ed5509568cdd9e4c387dcd60414e68dde740b54064e68ba8`. Existing verified output is reused; mismatched output is preserved and rejected. The download is about 344 MB and the export about 1.03 GB.
3. Set model names, reasoning efforts and operation timeout in the selected YAML. Full LME requires 500 complete histories. Full BEAM requires ten conversation owners with two questions in each of ten categories. A smaller selection must set `diagnostic: true` and must not be reported as full coverage.

Published configurations keep DeepSeek Flash with high answer/judge reasoning. For a separate scoring experiment, YAML accepts optional `answer_model` and `judge_model` overrides; selection uses the role override first, common YAML `model` second, and the existing deep/light environment value otherwise. Overrides affect scoring only; OAM extraction keeps its runtime configuration. Embedding requests use YAML `embedding_model` when provided, otherwise `OAMB_EMBEDDING_MODEL`. No names are compared or restricted. Both scoring roles share the private `LLM_BASE_URL` and `LLM_API_KEY`. Configure another endpoint in a separate client environment when needed. For example:

```yaml
answer_model: deepseek-flash
judge_model: deepseek-flash
answer_effort: adaptive
judge_effort: high
```

Each role accepts `none`, `low`, `high`, `max` or `adaptive`. `none` sends disabled thinking; `adaptive` sends adaptive thinking; neither sends `reasoning_effort`. Explicit levels retain that field. MiniMax request names additionally request separate reasoning and adaptive thinking, preserving the migrated adapter's behavior; supplier/model support must be verified for the chosen endpoint. Only the visible final response is judged. Native adaptive thinking does not replace the source-reviewed max-effort retry policy. Model names, endpoints and keys do not gate resume or completed-result reuse. OAMB does not strip provider prefixes, normalize aliases or decide whether two names denote the same model. It records the original names in plans and receipts; the operator decides model choice and result comparability. Request stage, prompt, reasoning and other non-model settings still identify accepted/unknown calls. Existing published profiles and historical evidence remain unchanged.
4. Use that full YAML with the same precheck/run commands and a fresh output directory. This starts a new evaluation; it does not reproduce or replace the historical scores.

The operator owns evaluation cost. OAMB records model and indexing usage without run-wide call/token limits or token reservations. Complete valid responses are reused unchanged; truncated, empty, refused or semantically invalid responses are retained and may receive a new attempt on explicit resume. A dispatch without a known HTTP outcome blocks redispatch until its receipt is reconciled. Model requests use the provider's default response allowance; truncated responses are rejected.

Resume the same run with:

```bash
./run.sh --oam outputs/oam-lme-example --resume
```

The first pass saves every answer and judgment before policy retries and generates `retry-eligibility.json` with one draft per failed question. LME failure means score below 1; BEAM failure means mean rubric score below 0.5. Review the original source, retrieved context, question, gold and rubric. For each draft, retain `context_sha256`, set `coverage_review_complete` and `gold_review_complete` to `true`, set `gold_status` to `correct`, `ambiguous` or `incorrect`, and write a nonempty `rationale`. Set `eligible: true` only when the substantive coverage and gold review allow an answer retry; incorrect gold requires `eligible: false`. A reviewed ineligible question retains its first-pass score and stays in the denominator.

Resume after completing those reviews. Each eligible question gets one max-effort answer using the identical prompt, context and model, then judging at the configured effort (high in published profiles). The accepted review is saved before dispatch and remains authoritative for that retry; later draft edits or deletion do not discard an admitted retry. Completed answers/judges are reused if execution stops between stages. A completed retry replaces the policy result even when its score is lower; the first pass is retained unchanged. `summary.json` and HTML show first-pass and policy-final means separately. Unreviewed failures appear in `pending_review`, leave `policy_final_mean_score` null, and print `FIRST PASS COMPLETE; REVIEW PENDING`; they are not a completed policy-final evaluation.

Resume checks the dataset, OAM scope/configuration/generation, prompt identity, non-model request settings and embedding dimension, reuses accepted source receipts and completed source/model/question results, and executes unfinished work. Harmless evaluator code edits do not block resume. If owner creation committed before its key was saved, root-authorized credential issuance recovers access to that same owner. It does not automatically retry a terminally failed indexing operation or treat a partial source as complete. Keep the isolated service and private environment files; stop/restart that exact Compose project when necessary. A stopped service's data is retained. Concurrent processes must not operate the same run directory.

Fresh precheck and resume require one finite embedding vector with the configured dimension. They do not compare embedding model names with runtime names or enforce a named model.

## 4. Inspect the result

Open `report.html` directly in a browser. `summary.json` records validated question coverage and mean score; `results/` records each exact retrieved context, its hash/token count, answer, judgments and score. `owners/` contains accepted source receipts and query-ready operation/usage records; user keys are separate owner-only files. `model-calls/` retains requests, supplier responses, finish reasons, elapsed time and reported usage, including reasoning. `raw/` correlates OAM requests, responses and elapsed time without authorization headers or issued secrets.

Final validation checks the exact selected question set, source-bound questions/golds/rubrics, the actual prompts and model outputs, complete recall diagnostics, context hashes and recomputed judge scores. A synthetic 1/1 result is a wiring check. These artifacts are a native OAM run, not a three-provider comparison capsule, matched quality study or release-acceptance pass. End-to-end retrieval time is retained in raw HTTP receipts; product stage timings remain in each recall response. Indexing usage belongs to the physical history and must not be multiplied by its number of questions.

## 5. Review saved context offline

Export a worksheet and complete prepared ingestion sources, including BEAM document-context wrappers, from a completed native run:

```bash
./run.sh --oam <run-directory> --export-review <review.json>
./run.sh --oam <run-directory> --review-report <review.json>
```

These modes read validated results without loading credentials or contacting OAM/model services. Export never overwrites a worksheet. Edit only each case's `assessment`: explain the task, review the sources, set completion flags and label context spans using UTF-8 byte offsets (`start` inclusive, `end` exclusive). Categories are `direct`, `related`, `unrelated`, `uncertain`, `unreviewed` and `format`; semantic labels require a review reason. Optional `evidence_requirements` link exact context quotes to source quotes by exported `session_id` and `message_index`. All source messages remain available regardless of gold annotations. Prepared sources reproduce the ingestion input, not a claim of byte-identical raw dataset text.

The companion `<review>.report.json` contains per-case counts and a token-weighted ratio over fully reviewed nonempty cases, with review coverage explicit. Counts encode the complete context once using native `o200k_base`; tokens crossing spans belong to the span containing their first non-whitespace byte. Missing/uncertain review yields no exact ratio; `measurement: lower_bound` reports a bound separately. Question/gold/context/result/source/tokenizer changes are rejected, and original answers, scores and receipts stay untouched. Prepare tokenizer cache assets separately; offline review never downloads them. Automated relevance estimation and live source-to-index diagnosis are outside this offline tool's claims.
