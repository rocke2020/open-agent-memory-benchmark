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

OAMB can reuse any local or paid OpenAI-compatible embedding service. The configured service must serve `qwen3-embedding:0.6b`, return 1,024 dimensions, and accept at least 8,192 input tokens. The API key is optional for an unauthenticated local service. If no embedding service URL is configured, macOS starts the checked-in vLLM-Metal helper and Linux starts the checked-in Ollama helper. The macOS fallback expects a sibling `vllm-metal` checkout and the cached model paths described by its error messages; the Linux fallback requires Ollama and may download the 639 MB model on first use. On Linux, Ollama binds to loopback and the checked-in bounded relay listens only on the Docker bridge gateway needed by provider containers; neither binds the unauthenticated service to LAN interfaces.

### 1. Precheck

Clone OAMB and enter its root:

```bash
git clone https://github.com/rocke2020/open-agent-memory-benchmark.git && \
  cd open-agent-memory-benchmark
```

Prepare the ignored root `.env` from `./.env.example`, keep `LLM_URL_TYPE=openai_chat`, set `LLM_BASE_URL` and `LLM_API_KEY`, and keep the file mode `0600`. To reuse a local or paid embedding API, set `OAMB_EMBEDDING_BASE_URL`; set `OAMB_EMBEDDING_API_KEY` only when that service requires authentication. You may omit `OAMB_EMBEDDING_API_KEY` or leave it empty for an unauthenticated service; both forms have the same behavior. For example, an already-running host-local vLLM-Metal service uses:

```dotenv
OAMB_EMBEDDING_BASE_URL=http://127.0.0.1:18000/v1
OAMB_EMBEDDING_API_KEY=
```

OAMB keeps this host-side URL unchanged for probing and derives `http://host.docker.internal:18000/v1` only for the provider containers. HTTPS or IPv6 loopback URLs are rejected because rewriting them would either break TLS hostname verification or leave containers pointing at themselves; use the HTTP IPv4 loopback form above or a hostname reachable from Docker. A paid service uses the same fields with its HTTPS base URL and API key.

Then run:

```bash
./precheck.sh
```

`configs/benchmark.yml` is the only source for model names, roles, thinking effort, environment-variable names, and the managed-local embedding host endpoint. Human-supplied external endpoints and credentials remain private in `.env`. `precheck.sh` completes private runtime inputs, freezes the embedding ownership and effective endpoint into the benchmark plan, and exports plan-owned settings to the provider processes. When the embedding URL is configured, it reuses that service and does not start vLLM-Metal or Ollama; a missing or empty API key selects keyless access. When the URL is absent or remains the documented placeholder, precheck starts the OS-specific fallback without changing `OAMB_EMBEDDING_BASE_URL` in `.env`. An API key without a URL fails closed. It then installs locked dependencies, downloads and verifies LongMemEval, clones and verifies the pinned Mem0 source, starts all three memory providers, and verifies every runtime role. Existing configured values and provider data are reused, not overwritten.

Provider applications derive their `NO_PROXY` list from the host in the root `LLM_BASE_URL`, together with the required local service hosts, so requests to that model endpoint bypass proxy routing.

The all-role readiness gate sends a real embedding request and validates the returned model and dimensions for either an external service or the local fallback. Its direct host-side probe sends no `Authorization` header when `OAMB_EMBEDDING_API_KEY` is missing or empty. For keyless endpoints, provider-owned OpenAI clients receive the non-secret `oamb-no-auth` compatibility value because some clients require a non-empty constructor argument. `--embedding-api-url URL` remains available as a one-run external URL override, does not require a key, and does not write the URL to `.env`. Use `./precheck.sh --no-start-embedding` only when the managed-local endpoint is already managed separately. Do not continue unless precheck prints `precheck: PASS`.

### 2. Run

Every smoke/full run first verifies the frozen plan and runtime readiness before
dispatch. To run only that same gate without model or provider calls, use
`./run.sh --dry-run`.

Run the complete balanced LME-60 comparison with:

```bash
./run.sh --full_test
```

Full mode starts Hindsight, Mem0, and OpenViking together with the default parallel limits. Each provider runs the same 60 questions in isolated state and reports its own progress from `0/60` through `60/60`. All three use up to 10 native extraction retries and three ingestion submissions per batch. After a settled batch exhausts its submissions, OAMB records the skipped sources, preserves any partial memory, and continues the question’s remaining history in the same scope. Answer and judge calls have independent limits of six outer attempts and two transport retries per attempt; invalid output is fed back with its validation error. The report identifies partial ingestion and unjudged results, and retains physical attempts and available usage evidence. OAMB freshly validates each capsule before building and opening the final 180-result comparison report. Full mode does not require or consume a smoke run.

Set parallel limits under `execution` in `configs/benchmark.yml` before running `./precheck.sh`:

```yaml
execution:
  max_parallel_providers_per_dataset: 3
  max_parallel_history_ingestions_per_provider: 2
  max_parallel_questions_per_provider: 2
```

These required positive integers limit active provider evaluations, independent history ingestions per provider, and question runs per provider. Each question slot spans retrieval, answer, and judge after its history is ready; source writes within one history stay sequential. The defaults allow up to six history ingestions and six question runs across three providers. A provider limit of 1 or 2 queues the remaining providers and starts the next when a slot becomes free. The limits apply to both smoke and full execution and are frozen into the resolved plan; editing YAML does not alter an existing plan. They do not limit a provider service's internal model or embedding requests.

Press `Ctrl-C` once to stop a full run. OAMB immediately force-stops the benchmark process tree and this repository's provider-service containers, while preserving atomically saved question results, provider volumes, and result data. Native workers also stop themselves if their `run.sh` owner or immediate supervisor disappears.

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

OAMB atomically saves every completed question; `--resume` skips existing question IDs and reruns only missing questions in fresh scopes.

Both full-test modes print each provider's status, elapsed time, and saved-question count at startup, every 30 seconds while running, and when the command finishes. Resume counts include previously saved results; providers already at `60/60` are shown as complete. The initial `reused=... remaining=...` line is a startup summary.

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
