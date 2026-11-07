# Native OAM evaluation workflows

## Overview

OAMB is OAM's standalone evaluation repository. Its native workflows run chronological interaction ingestion, operation polling, non-generative retrieval, answering and judging through the public `./precheck.sh` → `./run.sh` interface. They require Docker, Compose, Python 3.11+, `uv`, an OAM image and configured model endpoints. No product checkout, private runner or provider SDK is imported during evaluation. A separate PostgreSQL volume, API and worker belong to each deployment.

| Profile | Recall | Vector index | Question isolation | Scoring |
| --- | --- | --- | --- | --- |
| LME-500 | Fast | Shared `small` | One owner per question history | Existing OAMB LME category-specific yes/no judge |
| BEAM-10M | Normal | Dedicated `large` | One owner per conversation, reused by its questions | Mean of independent 0, 0.5 or 1 rubric-item scores |

Both profiles disable Insights, Profile and the planner and use low extraction reasoning. The full profiles are [LME-500](../configs/oam-lme500.yml) and [BEAM-10M](../configs/oam-beam10m.yml). The small [LME diagnostic](../configs/oam-lme500-smoke.yml) and [BEAM diagnostic](../configs/oam-beam10m-smoke.yml) use original synthetic fixtures; their results establish execution, not benchmark accuracy.

The reproduction target is the published evaluation procedure with the same models and scores close to the published results, allowing model variability and disclosed OAM version changes. The workflow includes the recorded LME templates and BEAM answer/nugget prompts, BEAM's structured JSON answer format and document-context metadata, and the reviewed max-effort retry procedure. It preserves conversation turns, roles, source dates, original questions and provider-returned context. Current OAM owns its extraction prompts, tokenizer and index behavior; record the evaluated revision and configuration. Keep new measurements separate from the saved [LME](evaluations/oam-v0.0.1-lme500.md) and [BEAM](evaluations/oam-v0.0.1-beam10m.md) campaigns. See [prompt provenance](../prompt-packs/README.md) and [the mainflow](designs/native-oam-reproduction.md).

## 1. Prepare the image and model environment

Install the locked dependencies and build the development image from committed OAM source:

```bash
uv sync --locked --all-groups
./provider-services/oam/build-image.sh /path/to/open-agent-memory 32f97cca5c5c1fd2546ed0364d2279063afc81b6
cp .env.oam.example .env.oam
chmod 600 .env.oam
```

Set `LLM_BASE_URL` and `LLM_API_KEY` in `.env.oam`. For the published evaluation's model setup, set both `LLM_LIGHT_MODEL` and `LLM_DEEP_MODEL` to the provider's identifier for **`deepseek-v4.1-flash`**. The native workflow uses OpenAI-compatible Chat Completions. The light binding performs extraction and judging; the deep binding answers, using the same model. If the endpoint uses a gateway alias such as `deepseek-flash`, verify that it resolves to the required model.

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
4. Use that full YAML with the same precheck/run commands and a fresh output directory. This starts a new evaluation; it does not reproduce or replace the historical scores.

The operator owns evaluation cost. OAMB records model and indexing usage without run-wide call/token limits or token reservations. Complete valid responses are reused unchanged; truncated, empty, refused or semantically invalid responses are retained and may receive a new attempt on explicit resume. A dispatch without a known HTTP outcome blocks redispatch until its receipt is reconciled. Model requests use the provider's default response allowance; truncated responses are rejected.

Resume the same run with:

```bash
./run.sh --oam outputs/oam-lme-example --resume
```

The first pass saves every answer and judgment before policy retries and generates `retry-eligibility.json` with one draft per failed question. LME failure means score below 1; BEAM failure means mean rubric score below 0.5. Review the original source, retrieved context, question, gold and rubric. For each draft, retain `context_sha256`, set `coverage_review_complete` and `gold_review_complete` to `true`, set `gold_status` to `correct`, `ambiguous` or `incorrect`, and write a nonempty `rationale`. Set `eligible: true` only when the substantive coverage and gold review allow an answer retry; incorrect gold requires `eligible: false`. A reviewed ineligible question retains its first-pass score and stays in the denominator.

Resume after completing those reviews. Each eligible question gets one max-effort answer using the identical prompt, context and model, then high-effort judging. The accepted review is saved before dispatch and remains authoritative for that retry; later draft edits or deletion do not discard an admitted retry. Completed answers/judges are reused if execution stops between stages. A completed retry replaces the policy result even when its score is lower; the first pass is retained unchanged. `summary.json` and HTML show first-pass and policy-final means separately. Unreviewed failures appear in `pending_review`, leave `policy_final_mean_score` null, and print `FIRST PASS COMPLETE; REVIEW PENDING`; they are not a completed policy-final evaluation.

Resume checks the dataset, model/runtime bindings, prompt identity and embedding dimension, reuses accepted source receipts and completed source/model/question results, and executes unfinished work. Harmless evaluator code edits do not block resume. If owner creation committed before its key was saved, root-authorized credential issuance recovers access to that same owner. It does not automatically retry a terminally failed indexing operation or treat a partial source as complete. Keep the isolated service and private environment files; stop/restart that exact Compose project when necessary. A stopped service's data is retained. Concurrent processes must not operate the same run directory.

Fresh precheck verifies the published embedding model setup and its agreement with OAM's extraction embedding binding. On resume, the embedding gate remains one finite vector with the frozen dimension; it adds no URL or alias-ownership audit.

## 4. Inspect the result

Open `report.html` directly in a browser. `summary.json` records validated question coverage and mean score; `results/` records each exact retrieved context, its hash/token count, answer, judgments and score. `owners/` contains accepted source receipts and query-ready operation/usage records; user keys are separate owner-only files. `model-calls/` retains requests, supplier responses, finish reasons, elapsed time and reported usage, including reasoning. `raw/` correlates OAM requests, responses and elapsed time without authorization headers or issued secrets.

Final validation checks the exact selected question set, source-bound questions/golds/rubrics, the actual prompts and model outputs, complete recall diagnostics, context hashes and recomputed judge scores. A synthetic 1/1 result is a wiring check. These artifacts are a native OAM run, not a three-provider comparison capsule, matched quality study or release-acceptance pass. End-to-end retrieval time is retained in raw HTTP receipts; product stage timings remain in each recall response. Indexing usage belongs to the physical history and must not be multiplied by its number of questions.
