# AGENTS.md

This file provides context for AI coding assistants (Claude Code, Codex, etc.)

## Project scope

Open Agent Memory Benchmark (OAMB) is the standalone evaluation repository for Open Agent Memory (OAM). OAM implements the memory system; OAMB owns dataset preparation, evaluation, execution and reporting. Use [README.md](README.md) for the public workflow and [docs/native-oam.md](docs/native-oam.md) for native OAM setup and execution.

Native OAM evaluates full LME-500 and BEAM-10M using the published procedure and models. The goal is comparable scores while allowing model variability and disclosed OAM changes. Preserve dataset identity, prompts, model roles, reasoning efforts, recall modes, scoring and retry rules; record the evaluated OAM revision and configuration.

The separate REST comparison evaluates Hindsight, Mem0 and OpenViking on balanced LME-60: ten questions from each of six types, producing 180 provider-specific results. Its settings and artifacts are separate from native OAM. Mem0's optional SDK profile is non-default and cannot substitute for REST evidence.

Published results and their provenance belong in [the LME report](eval_results/lme500/README.md) and [the BEAM report](eval_results/beam10m/README.md). Preserve those historical campaigns; new evaluations produce separate outputs. Do not rerun a completed campaign merely to update documentation.

## Public workflows

The supported evaluation interface is `./precheck.sh` followed by `./run.sh`. Package-internal commands implement these scripts and are not an alternative public workflow. Service preparation and dataset download helpers are documented in the README and native guide.

For native OAM, prepare an isolated deployment and its private client environment, then run:

```bash
./precheck.sh --oam configs/oam-lme500.yml --env-file <client.env> --output <run-directory>
./run.sh --oam <run-directory>
./run.sh --oam <run-directory> --resume
```

Use `configs/oam-beam10m.yml` for BEAM, or the corresponding `*-smoke.yml` for a synthetic diagnostic. Full datasets come from `scripts/download/longmemeval.sh` and `scripts/download/beam10m.sh`; downloaded data stays outside Git. A smaller question/history selection must be identified as a diagnostic, not full-dataset coverage.

For the separate comparison, use `./precheck.sh` and `./run.sh --full_test`; add `--resume` to continue unfinished work. `configs/benchmark.yml` owns comparison settings. Optional smoke, dry-run and saved-result reporting modes are documented in the README.

Keep secrets and machine-local endpoints in private environment files. Native settings live in `configs/oam-*.yml`, with OAM runtime overrides in `provider-services/oam/`. Do not edit a saved plan to change an evaluation already in progress.

## Evaluation behavior

- Model names and defaults belong in the selected configuration and run evidence; model/provider choice and result comparability belong to the operator. Published profiles use low extraction reasoning, high initial answer/judge reasoning and max reasoning for an eligible answer retry. Scoring supports `none`/`low`/`high`/`max`/`adaptive`; preserve the selected profile and record resolved roles. Native adaptive thinking and the reviewed max-retry policy are separate behaviors.
- Embedding model and dimensions come from the selected configuration. Record tokenizer or embedding changes because they can affect chunking and results. On resume, verify one finite vector with the recorded dimension; do not add endpoint-ownership or embedding-migration audits.
- Preserve chronological source order, roles, timestamps, original questions, golds and rubrics. Wait for complete extraction and query readiness before evaluating that history. Partial extraction is not successful ingestion even when the operation status is `succeeded`.
- LME uses fast recall with the shared `small` index; BEAM uses normal recall with `large` per-owner indexes. Both disable Insights, Profile and planning. Provision BEAM's owner index before retrieval.
- Retrieval is non-generative. Query embedding is allowed; query rewriting, reflection, generative reranking and other generative memory-query operations are prohibited. Preserve provider-returned retrieval order; OAMB adds no reranker.
- Save first-pass answers and judgments before retries. A failed question may receive one source/context/gold-reviewed answer retry on the same prompt, context and model. Incorrect gold blocks retry. The completed retry becomes policy-final even when its score is lower; keep first-pass and policy-final results separately.
- Resume reuses accepted ingestion receipts and completed model/question results and runs unfinished work. Preserve failed and ambiguous attempts; do not silently redispatch a model call with an unknown outcome or operate the same run directory concurrently.
- Treat model names as opaque user configuration. Do not normalize aliases, infer equivalence, enforce model-name agreement or block resume on names/endpoint/key changes. Pass chosen strings unchanged and record provenance; the operator decides comparability. Preserve completed work and unknown-outcome guards using stage and non-model request settings.
- Keep accuracy, context tokens, indexing tokens, retrieval latency and indexing time separate. BEAM mean rubric score differs from the fraction of questions meeting its threshold. Missing measurements remain unavailable. Attribute indexing usage to the physical history, not every question sharing it.

## Codes Architecture

The native execution path is `precheck.sh` / `run.sh` → `oam_workflow.py` → workload loading, OAM HTTP operations, model calls and saved results. The separate comparison is composed through `cli.py`, `live.py` and `runtime/`.

| Path | Responsibility |
| --- | --- |
| `src/oamb/oam_workflow.py` | Native precheck, chronological execution, checkpoints, retry integration, validation and HTML/JSON results |
| `src/oamb/workloads/` | Dataset loading, published prompt templates, model request formats, scoring and retry-review rules |
| `src/oamb/workloads/model_parameters.py`, `context_noise.py`, `oam_review.py` | Migrated reasoning adaptation and offline source/span review; no dependency on the private OAM evaluation checkout |
| `src/oamb/memory_systems/` | Provider adapters; `oam.py` owns native OAM HTTP operations and request packing |
| `src/oamb/model_clients/` | Comparison model transports; optional clients load only when selected |
| `src/oamb/runtime/` | Comparison execution, attempts, readiness, resume and lifecycle |
| `src/oamb/contracts/` | Typed identities, records, ports and artifact parsing; no concrete runtime, adapter, model-client or storage imports |
| `src/oamb/artifacts/` and `src/oamb/reporting/` | Saved-artifact validation, comparison reduction and offline reports |
| `src/oamb/external_evidence/` | Read-only historical evidence; separate from native execution artifacts |
| `src/oamb/public_boundary.py` | Public-artifact checks for local-only paths and symlinks |
| `provider-services/` | Isolated service preparation, image/source pins and non-destructive service management; never imported by the Python package |
| `scripts/download/` | Pinned input acquisition and verified installation |

Concrete adapters do not import runtime or each other. Keep core import, command help, artifact parsing and validation free of provider SDKs, databases, model clients and credentials. Use one owner for each configuration or serialization rule; avoid generic dumping-ground modules.

## Development and verification

Use Python 3.11+ and the committed `uv.lock`:

```bash
uv sync --locked --all-groups
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest -q
python3 -m unittest discover -s provider-services/tests -p 'test_*.py' -v
uv run python scripts/check_public_boundary.py
uv build
```

Implement behavior test-first and run the affected checks. Tests must not make paid calls, write/delete real provider or database data, or perform destructive container operations. Validator tests need a valid case and an independently invalid case that proves rejection.

Verify the actual affected path before claiming it works. A native diagnostic covers precheck, isolated ingestion, readiness, retrieval, answer, judge, saved evidence and inspectable HTML. Mocks and synthetic samples prove their own boundaries; they do not establish full-dataset accuracy, historical-score parity or release readiness.

Keep provider/database data and partial, failed or interrupted artifacts. Never expose credentials, private content or resolved secrets in logs or public validation messages. Store disposable verification output in `tmp/` and persistent local evidence in `personal/`; keep bulk/generated data outside Git.

## Change discipline

Make the smallest complete change for the requested evaluation behavior and preserve unrelated work. Reuse existing checkpoints, adapters and reporting rather than adding parallel mechanisms. Native OAM records usage without harness spending limits or token reservations; the operator controls cost.

Keep README commands, configuration, prompts, design documentation and their actual consumers aligned. Before release, update typed artifact producers, parsers and consumers together; add compatibility machinery only for a demonstrated need to reopen existing public evidence.

Report implementation, offline verification, live evidence, full-dataset quality, commit and push status separately. A successful check, finalized run or validated artifact does not imply a high score or a completed quality evaluation.
