# Open Agent Memory Benchmark

## What is Open Agent Memory Benchmark?

Open Agent Memory Benchmark (OAMB) is an open-source, provider-neutral way to
compare agent-memory systems under one shared evaluation protocol. Agent memory
is the layer that turns an agent's past interactions into searchable memory and
retrieves useful information for later tasks.

The v0.1.0 profile runs Hindsight, Mem0, and OpenViking on the same balanced selection of 60 LongMemEval questions: ten questions from each of six question types. It reports five main metrics separately, in decision-priority order:

- **Answer accuracy** — how often the agent answers correctly using retrieved memory.
- **Context tokens** — how much retrieved memory is shown to the answer model.
- **Indexing tokens** — the measured token usage required to turn the test history into searchable memory.
- **Retrieval latency** — how long the provider memory-query request takes.
- **Indexing time** — how long the complete ingestion path takes from the first provider write through terminal readiness.

Retries, failures, resource usage, cost, and measurement coverage remain visible
supporting evidence. OAMB does not hide the trade-offs inside one universal
score, so users can choose a memory system according to their own needs.

## Why Open Agent Memory Benchmark?

Most agent-memory comparisons are published by individual memory providers.
Their results are hard to compare because they may use different prompts,
extraction, answer, and judge models, retrieval modes, limits, or token
definitions. Important adoption costs, especially indexing tokens, may be
missing entirely.

OAMB fixes the questions, answer and judge policy, model roles, and comparison
rules before a run starts. Each provider still uses its native memory API, while
provider-specific settings and measured results remain visible in saved evidence
and a self-contained offline report.

OAMB separates memory extraction, answer generation, and scoring. Its extraction
context frames each session as a conversation in which the model is the
assistant, preserves the session timestamp, and omits the benchmark name. Answer
generation uses OAMB's evidence-grounded prompt. Scoring preserves the original
LongMemEval judge rubrics, with source attribution and byte-pinned templates.

The goal is a fair and open evaluation of agent memory that is quick to run,
inspect, and reproduce.

## How do I use Open Agent Memory Benchmark?

The normal workflow is two commands: `precheck.sh` prepares and verifies the
complete runtime, then `run.sh --full_test` runs all three providers in
parallel, validates their capsules, builds the comparison, and opens its offline
report. Smoke mode is an optional one-question diagnostic.

`execution.max_retries_per_operation` in `configs/benchmark.yml` is fixed at `2` additional batch submissions, for three total attempts after up to 10 native extraction retries per submission. A settled batch that exhausts those attempts is recorded as skipped, and the remaining history continues in the same question scope. Answer and judge each use six outer attempts with two transport retries per attempt. Every physical attempt's available token usage, resource measurements, and actual billing evidence remain in the operational totals; see the [runtime retry flow](src/oamb/runtime/native_run.py).

LongMemEval [visible evidence](src/oamb/workloads/visible_evidence.py) preserves provider-returned empty text in its original position and JSONL representation. The [report parser](src/oamb/reporting/comparison_project.py) accepts the same representation; evidence identity and kind must remain non-empty.

LongMemEval source messages also preserve empty string content through the [Mem0 adapter](src/oamb/memory_systems/mem0/adapter.py) and [REST request encoder](src/oamb/memory_systems/mem0/wire.py), matching the dataset and native service schema. Message roles remain non-empty strings, and non-string content is rejected.

The 900-second timeout applies to each external attempt. These controls do not impose a whole-run, aggregate-token, storage, resource, memory, or cost cap. Tokens, latency, storage, resources, and cost are measured results; unavailable measurements remain unavailable rather than becoming zero.

Retrieval generation is disabled in v0.1.0. Query embedding is allowed, but
query rewriting, decomposition, reflection, generative reranking, and fallback
to a generation-model retrieval path are not.

## Quick start

These two steps run a real comparison. They create isolated provider state and
can make billable model calls; neither script deletes provider or database data.

Install Git, Python 3.11 or newer, `uv`, Docker Engine with Compose, `curl`,
`jq`, and `shasum`. You also need an OpenAI-compatible endpoint that serves
`deepseek-v4-flash` and `deepseek-v4-pro`.

For local embeddings, macOS uses the checked-in vLLM-Metal helper and Linux uses
the checked-in Ollama helper. The embedding model must be
`qwen3-embedding:0.6b`, return 1,024 dimensions, and accept at least 8,192 input
tokens. The macOS helper expects a sibling `vllm-metal` checkout and the cached
model paths described by its error messages; the Linux helper requires Ollama
and may download the 639 MB model on first use. On Linux, Ollama binds only to
the Docker bridge gateway needed by the provider containers, not to LAN interfaces.

### 1. Precheck

Clone OAMB and enter its root:

```bash
git clone https://github.com/rocke2020/open-agent-memory-benchmark.git && \
  cd open-agent-memory-benchmark
```

Prepare the ignored root `.env` from `./.env.example`, keep
`LLM_URL_TYPE=openai_chat`, set `LLM_BASE_URL` and `LLM_API_KEY`, and keep the
file mode `0600`. Then run:

```bash
./precheck.sh
```

`configs/benchmark.yml` is the only source for model names, roles, and thinking
effort; those values are never copied into `.env`. `precheck.sh` completes the
private endpoint, credential, and host-runtime inputs in `.env`, freezes the
benchmark plan, and exports its model settings to the provider processes. It
then installs locked dependencies, downloads and verifies LongMemEval, clones
and verifies the pinned Mem0 source, starts the OS-specific embedding server
when needed, starts all three memory providers, and verifies every runtime role.
Existing configured values and provider data are reused, not overwritten.

Provider applications derive their `NO_PROXY` list from the host in the root `LLM_BASE_URL`, together with the required local service hosts, so requests to that model endpoint bypass proxy routing.

To use an OpenAI-compatible online embedding endpoint directly, pass its base
URL. The endpoint must accept the profile's `oamb-local-embedding` bearer value:

```bash
./precheck.sh --embedding-api-url https://embedding.example/v1
```

To keep an already configured embedding endpoint without starting a local
server, use `./precheck.sh --no-start-embedding`. In both cases the all-role
readiness gate still sends a real embedding request and validates the returned
model and dimensions. Do not continue unless precheck prints `precheck: PASS`.

### 2. Run

Every smoke/full run first verifies the frozen plan and runtime readiness before
dispatch. To run only that same gate without model or provider calls, use
`./run.sh --dry-run`.

Run the complete balanced LME-60 comparison with:

```bash
./run.sh --full_test
```

Full mode immediately starts Hindsight, Mem0, and OpenViking together. Each provider runs the same 60 questions in isolated state and reports its own progress from `0/60` through `60/60`. All three use up to 10 native extraction retries and three ingestion submissions per batch. After a settled batch exhausts its submissions, OAMB records the skipped sources, preserves any partial memory, and continues the question’s remaining history in the same scope. Answer and judge calls have independent limits of six outer attempts and two transport retries per attempt; invalid output is fed back with its validation error. The report identifies partial ingestion and unjudged results, and retains physical attempts and available usage evidence. OAMB freshly validates each capsule before building and opening the final 180-result comparison report. Full mode does not require or consume a smoke run.

Press `Ctrl-C` once to stop a full run. OAMB immediately force-stops the benchmark process tree and this repository's provider-service containers, while preserving atomically saved progress, provider volumes, and result data. Native workers also stop themselves if their `run.sh` owner or immediate supervisor disappears.

After deterministic comparison data closes, report generation starts two bounded branches: base HTML preparation and one concise five-metric analysis using the frozen judge model and the configured `LLM_BASE_URL`/`LLM_API_KEY`. Analysis permits at most six total attempts, stores received responses in a content-addressed private cache, and publishes `report-analysis.json` only after strict schema and report-hash validation. Final HTML waits for both branches and embeds a valid analysis directly; if analysis remains unavailable, the numeric report still opens with an explicit unavailable notice. A matching cache prevents another billable call, and the final HTML remains self-contained and network-free. Analysis-call usage is recorded in the sidecar but remains outside the five benchmark metrics.

For optional debugging, run the frozen question `72e3ee87` once on all three
providers in parallel. `--smoke_test` is the default, so these are identical:

```bash
./run.sh
# ./run.sh --smoke_test
```

Smoke freshly validates all three capsules, builds a diagnostic comparison over
the same question, and opens `report.html`. Its one-question report never claims
a full-study accuracy leader, and its artifacts are not a full-run prerequisite.

If an explicitly started full run stops after sealing partial capsules, resume
it with:

```bash
./run.sh --full_test --resume
```

Resume strictly loads `results/progress-hindsight.json`, `results/progress-mem0.json`, and `results/progress-openviking.json` before any provider or model construction. Missing, malformed, duplicate-key, non-finite, or identity-drifted progress fails closed without retrying the parse or treating it as empty. Each provider reuses completed question results unchanged, runs only unfinished questions in a fresh isolated scope, and atomically publishes each new terminal result. Once all 180 provider-question results are terminal, comparison is rebuilt directly from the three progress files. Resume never applies to smoke mode and never falls back to a fresh full run.

Smoke reports are written under `outputs/smoke-test/<run-label>/`; full-study
reports are written under `outputs/full-test/<run-label>/`. Set
`OAMB_NO_OPEN=1` only in a headless environment; the HTML is still built and
validated, and its path is printed.

`run.sh` saves stdout and stderr to a new private `outputs/tmp/run-<mode>-<UTC-timestamp>-<pid>.log` while continuing to display them in the terminal. The script prints `run: log=<path>` at startup; use `tail -f <path>` to follow it from another terminal. Logging includes preflight errors and applies to smoke, full, dry-run, and resume invocations; earlier logs are preserved.

## If a run stops or fails

Preserve `outputs/smoke-test`, `outputs/full-test`, `provider-services/.runtime`,
and all provider state. Do not delete or overwrite them. Inspect the printed
result-map paths first; they retain canonical outcomes and every completed
capsule root. `outputs/tmp` contains shared preparation state and reproducible
scratch outputs, not successful-run evidence.

Preserve the three canonical progress files and all diagnostic capsule roots. A later `--resume` reloads progress under one process lock, reports the exact reused and remaining question counts, and dispatches only providers with unfinished questions. If progress is missing, malformed, modified, or incompatible with the current plan and frozen case manifest, resume stops before external dispatch.

A readiness failure whose runtime is already bound to a provider project needs
a separate fresh clone and provider project. Stop the old project without
deleting its volumes:

```bash
./provider-services/bin/provider-services stop
```

The per-operation timeout limits local waiting for one attempt. It does not
prove remote work or supplier billing stopped, so reconcile uncertain calls
before starting another project.

## Provider and evidence boundaries

The default comparison uses local REST APIs for Hindsight, Mem0, and
OpenViking. The optional Mem0 Python SDK profile is separate and cannot
substitute for REST evidence. Exact release pins, service preparation, state
isolation, and non-destructive lifecycle commands are documented in
[`provider-services/README.md`](provider-services/README.md).

Source inspection, a healthy service, or a non-empty response alone does not
prove a completed benchmark. OAMB comparison reads freshly validated capsules
and reports missing measurements as unavailable.

## Project policies

- [`SECURITY.md`](SECURITY.md) — vulnerability reporting and secret safety.
- [`DATASETS.md`](DATASETS.md) — dataset provenance and redistribution rules.
- [`THIRD_PARTY.md`](THIRD_PARTY.md) — direct dependency and license inventory.

OAMB is licensed under Apache-2.0. Datasets and third-party artifacts retain
their own licenses and are not relicensed by OAMB.
