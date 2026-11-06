# Native OAM evaluation workflows

## Overview

The native OAM workflows run chronological interaction ingestion, operation polling, non-generative retrieval, answering and judging through the public `./precheck.sh` → `./run.sh` interface. They require Docker, Compose, Python 3.11+, `uv`, an OAM image and configured model endpoints. No product checkout, private runner or provider SDK is imported during evaluation. A separate PostgreSQL volume, API and worker belong to each deployment.

| Profile | Recall | Vector index | Question isolation | Scoring |
| --- | --- | --- | --- | --- |
| LME-500 | Fast | Shared `small` | One owner per question history | Existing OAMB LME category-specific yes/no judge |
| BEAM-10M | Normal | Dedicated `large` | One owner per conversation, reused by its questions | Mean of independent 0, 0.5 or 1 rubric-item scores |

Both profiles disable Insights, Profile and the planner and use low extraction reasoning. The full profiles are [LME-500](../configs/oam-lme500.yml) and [BEAM-10M](../configs/oam-beam10m.yml). The small [LME diagnostic](../configs/oam-lme500-smoke.yml) and [BEAM diagnostic](../configs/oam-beam10m-smoke.yml) use original synthetic fixtures; their results establish execution, not benchmark accuracy.

These are new OAMB protocols. BEAM uses OAMB-owned answer and rubric prompts. The workflows do not implement the historical campaign's max-effort answer-retry policy, inject document-context metadata or reproduce its tokenizer settings. They preserve conversation turns, roles, source dates, original questions and provider-returned context. Keep their results separate from the saved [LME](evaluations/oam-v0.0.1-lme500.md) and [BEAM](evaluations/oam-v0.0.1-beam10m.md) campaigns.

## 1. Prepare the image and model environment

Install the locked dependencies and build the development image from committed OAM source:

```bash
uv sync --locked --all-groups
./provider-services/oam/build-image.sh /path/to/open-agent-memory d41fd439956a7ec4e92d0d67185250be4e86ee35
cp .env.example .env.oam
chmod 600 .env.oam
```

Set `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_LIGHT_MODEL` and `LLM_DEEP_MODEL` in `.env.oam`. The native workflow uses OpenAI-compatible Chat Completions. The light model performs extraction and judging; the deep model answers. Also set `OAMB_EMBEDDING_BASE_URL`, optional `OAMB_EMBEDDING_API_KEY`, and `OAMB_EMBEDDING_MODEL`. The supplied OAM overrides expect 1,024-dimensional embeddings. Adjust the runtime override and benchmark's `embedding_dimensions` together for another dimension.

Use model endpoints reachable from both the host and the containers. The preparation helper maps loopback endpoints to `host.docker.internal`, as verified on Docker Desktop. On Linux, provide a container-reachable endpoint or configure the host-gateway mapping. Linux host-local model routing has not been live-verified here. Endpoints and keys remain together in private environment files; never commit those files.

The image pins OAM source and its lockfile. Its base image and OS packages are not fully content-pinned, so the build does not promise identical image bytes. Current ingestion uses OAM's configured tokenizer; match the embedding model's own tokenizer and input/request limits before a large run. Changing these settings does not reproduce historical chunking.

## 2. Start an isolated service and run a small sample

Use a fresh output directory and Compose project name. The setup helper rejects existing project containers or its retained PostgreSQL data volume before any database mutation, creates separate non-superuser content/API/queue login roles, migrates the new database, bootstraps credentials and writes owner-only environment files. It selects a fixed free loopback API port and waits up to 60 seconds for API readiness before publishing the client environment. There are no deletion or cleanup commands.

```bash
uv run --locked python provider-services/oam/start.py \
  --dataset lme500 --env-file .env.oam \
  --image oamb/oam:0.0.1-d41fd439956a \
  --project oamb-native-lme-example \
  --output outputs/oam-services/lme-example

./precheck.sh --oam configs/oam-lme500-smoke.yml \
  --env-file outputs/oam-services/lme-example/client.env \
  --output outputs/oam-lme-example
./run.sh --oam outputs/oam-lme-example
```

For BEAM, start a second fresh deployment with `--dataset beam10m`, a new project/output name, and use `configs/oam-beam10m-smoke.yml` in precheck. Do not switch a running deployment's dataset override. The generated client file includes the actual bootstrap tenant; tenant names are not assumed to be `default`.

Precheck verifies the live API contract, resolved in-container configuration hash and profile, and one finite embedding vector of the configured dimension. It freezes dataset bytes, selected questions, model bindings, runtime identities, budgets and prompt identity in `plan.json`. It performs no generation calls. Run repeats runtime and embedding checks before writing an isolated user collection. The dataset and credentials must remain available at their recorded paths.

Run preserves source order, roles and timestamps and splits requests only to satisfy the API's message/UTF-8 byte limits. Every batch occurrence has its own stable idempotency key. It requires successful source operations with complete extraction, source/memory readiness and semantic/temporal link readiness; partial extraction fails even when the operation reports success. Failed and ambiguous artifacts remain available.

After indexing, BEAM provisions each exact conversation owner's dedicated index. Both profiles execute and explain OAM's actual dense-discovery SQL under its configured planning mode, using a stored owner vector and the content role. A missing expected shared/dedicated index in that plan stops execution. This proves the diagnostic query's selection; it is not a claim that every future query or workload has the same plan or latency.

## 3. Full datasets, budgets and resume

No full evaluation is required to verify the adapter. To intentionally start a future full run, supply the inputs and review the ceilings before choosing the corresponding full configuration:

1. Download LME using `./scripts/download/longmemeval.sh`; its full profile uses the pinned 500-question source.
2. Supply BEAM-10M's JSON export at `datasets/beam/10m.json`, retaining its nested conversation/plan/batch/turn and probing-question fields. The full profile identifies the recorded input by SHA-256 `c16643a7cc7d93c1ed5509568cdd9e4c387dcd60414e68dde740b54064e68ba8`, from dataset revision `9b2096193fe74e2837e4713e483351e19817773c`. The export is not distributed or automatically downloaded. A differently serialized export requires an explicitly updated source hash before precheck; it is a new input identity.
3. Review model names, reasoning efforts, output allowance, operation timeout and budgets in the selected YAML. Full LME requires 500 complete histories. Full BEAM requires ten conversation owners with two questions in each of ten categories. A smaller selection must set `diagnostic: true` and must not be reported as full coverage.
4. Use that full YAML with the same precheck/run commands and a fresh output directory. This starts a new evaluation; it does not reproduce or replace the historical scores.

`max_model_calls` and `max_model_tokens` bound answer/judge dispatches, including failed, interrupted and resumed attempts. Every dispatch reserves its allowance before the HTTP call. Complete valid responses are reused unchanged; truncated, empty, refused or semantically invalid judge responses are retained and may receive a new attempt only on an explicit resume within the same allowance. There are no automatic harness retries. An unknown billed outcome retains its reservation. The output cap includes reasoning and the complete visible body and is never used to trim content.

`max_indexing_tokens` separately bounds generation triggered by ingestion. OAM's `index.generation_token_budget` enforces a per-source ceiling internally. The serial runner reserves that full source ceiling before dispatch; after successful completion, complete reported generation usage replaces that reservation. Missing/partial usage retains the ceiling. Embedding tokens are excluded. A budget failure stops before another source or model dispatch; do not edit a frozen plan to bypass it.

Resume the same run with:

```bash
./run.sh --oam outputs/oam-lme-example --resume
```

Resume checks the dataset, model/runtime bindings, prompt identity and embedding dimension, reuses accepted source receipts and completed source/model/question results, and executes unfinished work. If owner creation committed before its key was saved, root-authorized credential issuance recovers access to that same owner. It does not automatically retry a terminally failed indexing operation or treat a partial source as complete. Keep the isolated service and private environment files; stop/restart that exact Compose project when necessary. A stopped service's data is retained. Concurrent processes must not operate the same run directory.

## 4. Inspect the result

Open `report.html` directly in a browser. `summary.json` records validated question coverage and mean score; `results/` records each exact retrieved context, its hash/token count, answer, judgments and score. `owners/` contains accepted source receipts, query-ready operation/usage records and actual index plans; user keys are separate owner-only files. `model-calls/` retains requests, supplier responses, finish reasons, elapsed time and reported usage, including reasoning. `raw/` correlates OAM requests, responses and elapsed time without authorization headers or issued secrets.

Final validation checks the exact selected question set, source-bound questions/golds/rubrics, the actual prompts and model outputs, complete recall diagnostics, context hashes and recomputed judge scores. A synthetic 1/1 result is a wiring check. These artifacts are a native OAM run, not a three-provider comparison capsule, matched quality study or release-acceptance pass. End-to-end retrieval time is retained in raw HTTP receipts; product stage timings remain in each recall response. Indexing usage belongs to the physical history and must not be multiplied by its number of questions.
