# Open Agent Memory Benchmark

## What is Open Agent Memory Benchmark?

Open Agent Memory Benchmark (OAMB) is an open-source, provider-neutral way to
compare agent-memory systems under one shared evaluation protocol. Agent memory
is the layer that turns an agent's past interactions into searchable memory and
retrieves useful information for later tasks.

The v0.1.0 profile runs Hindsight, Mem0, and OpenViking on the same balanced
selection of 60 LongMemEval questions: ten questions from each of six question
types. It reports four main metrics separately:

- **Answer accuracy** — how often the agent answers correctly using retrieved
  memory.
- **Context tokens** — how much retrieved memory is shown to the answer model.
- **Indexing tokens** — the measured token usage required to turn the test
  history into searchable memory.
- **Latency** — how long indexing readiness and memory retrieval take.

Retries, failures, resource usage, cost, and measurement coverage remain visible
supporting evidence. OAMB does not hide the trade-offs inside one universal
score, so users can choose a memory system according to their own needs.

## Why Open Agent Memory Benchmark?

Most agent-memory comparisons are published by individual memory providers.
Their results are hard to compare because they may use different questions,
prompts, answer and judge models, retrieval modes, limits, or token definitions.
Important adoption costs, especially indexing tokens, may be missing entirely.

OAMB fixes the questions, answer and judge policy, model roles, and comparison
rules before a run starts. Each provider still uses its native memory API, while
provider-specific settings and measured results remain visible in saved evidence
and a self-contained offline report.

The goal is a fair and open evaluation of agent memory that is convenient to
run, inspect, and reproduce.

## How do I use Open Agent Memory Benchmark?

OAMB uses one explicit workflow:

1. Configure the workload, providers, model roles, and two runtime controls in
   `configs/benchmark.yml`.
2. Run `oamb doctor` to validate the configuration and freeze an immutable
   resolved plan before any provider call.
3. Prepare and verify the exact-pinned local provider services.
4. Run one frozen question through each provider and validate all three saved
   capsules. This is the small real-use gate before the larger evaluation.
5. Run the 60-question comparison, then freshly validate every full capsule.
6. Run `oamb compare` to build all three pairwise comparisons and one offline
   HTML report.

The checked-in runtime controls allow two additional safe attempts per eligible
operation and a 900-second timeout per external attempt. They do not impose a
whole-run, aggregate-token, storage, resource, memory, or cost cap. Tokens,
latency, storage, resources, and cost are measured results; unavailable
measurements are reported as unavailable, never as zero.

Retrieval generation is disabled in v0.1.0. Query embedding is allowed, but
query rewriting, decomposition, reflection, generative reranking, and fallback
to a generation-model retrieval path are not.

## Quick start

This walkthrough starts from a Git clone and performs a real LME-60 evaluation.
It writes isolated provider state and makes potentially billable model calls.
Run every command from the repository root and keep using the same shell so the
working variables remain available.

Before starting, install Git, Python 3.11 or newer, `uv`, Docker Engine with
Compose, `curl`, `jq`, and `shasum`. You also need:

- an OpenAI-compatible DeepSeek endpoint that provides `deepseek-v4-flash` and
  `deepseek-v4-pro`;
- credentials for that endpoint; and
- an OpenAI-compatible embedding endpoint serving
  `qwen3-embedding:0.6b` with 1,024 dimensions and at least an 8,192-token model
  and scheduler batch limit.

On Apple Silicon, the official
[vLLM-Metal installation guide](https://github.com/vllm-project/vllm-metal/blob/main/docs/installation.md)
and
[embedding guide](https://github.com/vllm-project/vllm-metal/blob/main/docs/text_embedding_pooling.md)
provide one way to run the embedding endpoint. After installing vLLM-Metal,
this command requests the required public model name and limits on port `18000`;
the later provider preflight verifies the actual response:

```bash
source ~/.venv-vllm-metal/bin/activate

VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_METAL_USE_PAGED_ATTENTION=1 \
VLLM_METAL_MEMORY_FRACTION=auto \
vllm serve mlx-community/Qwen3-Embedding-0.6B-8bit \
  --runner pooling \
  --served-model-name qwen3-embedding:0.6b \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --port 18000
```

Keep that server running in its own terminal.

### 1. Clone OAMB and prepare its inputs

```bash
set -euo pipefail

git clone https://github.com/rocke2020/open-agent-memory-benchmark.git
cd open-agent-memory-benchmark

uv sync --locked --group download
./scripts/download/longmemeval.sh

mkdir -p .local-demo/provider-source
git clone --branch v2.0.19 --single-branch \
  https://github.com/mem0ai/mem0.git \
  .local-demo/provider-source/mem0

test "$(git -C .local-demo/provider-source/mem0 \
  rev-parse 'v2.0.19^{commit}')" = \
  "dc82354e143c2581d505d581a00286d6ef8c3605"
```

The dataset downloader verifies the pinned revision and SHA-256. The final
`test` command independently verifies the Mem0 release commit and stops on a
mismatch.

Create one unique working directory for this attempt:

```bash
OAMB_RUN_LABEL="lme60-$(date -u +%Y%m%d-%H%M%S)"
OAMB_WORK_DIR="$PWD/.local-demo/$OAMB_RUN_LABEL"
MEM0_CHECKOUT="$PWD/.local-demo/provider-source/mem0"

mkdir -p \
  "$OAMB_WORK_DIR/bounded" \
  "$OAMB_WORK_DIR/full" \
  "$OAMB_WORK_DIR/results" \
  "$OAMB_WORK_DIR/validations"
```

### 2. Configure credentials and local services

```bash
cp .env.example .env
chmod 600 .env

cp provider-services/.env.example provider-services/.env
chmod 600 provider-services/.env

printf 'OAMB_PROVIDER_PROJECT=oamb-providers-%s\n' "$OAMB_RUN_LABEL"
printf 'OAMB_MEM0_SOURCE_CHECKOUT=%s\n' "$MEM0_CHECKOUT"
```

Open `.env` in a text editor and replace both placeholders:

```text
DEEPSEEK_BASE_URL=https://your-openai-compatible-endpoint/v1
DEEPSEEK_API_KEY=your-api-key
```

Then open `provider-services/.env` and:

- replace `OAMB_PROVIDER_PROJECT` and `OAMB_MEM0_SOURCE_CHECKOUT` with the two
  values printed above;
- replace every `change-me` value;
- set the three provider-model endpoints and credentials; and
- keep the pinned model names, ports, and embedding model unchanged unless your
  embedding server uses a different reachable host URL.

The provider dotenv parser intentionally does not support quotes, shell
expansion, backticks, backslashes, or inline comments. Use one plain
`NAME=value` per line.

### 3. Freeze the plan and verify every runtime role

```bash
uv run --locked oamb doctor \
  configs/benchmark.yml \
  --output "$OAMB_WORK_DIR/plan"

PLAN="$OAMB_WORK_DIR/plan/resolved-plan.json"

./provider-services/bin/provider-services doctor
./provider-services/bin/provider-services build
./provider-services/bin/provider-services up
./provider-services/bin/provider-services verify --services
```

The next command is the first potentially billable step. It probes the shared
embedding role, all three provider-owned producer roles, the answer role, the
judge role, and OpenViking readiness once each. It also proves that
provider-internal retries are disabled.

```bash
./provider-services/bin/provider-services verify --model-readiness \
  --resolved-plan "$PLAN" \
  --model-env "$PWD/.env"

./provider-services/bin/provider-services status
```

Do not continue unless both verification commands report `PASS`.

### 4. Run and validate one question on every provider

Use the same frozen question for all three providers:

```bash
uv run --locked oamb run "$PLAN" \
  --cell hindsight-lme60 \
  --question 72e3ee87 \
  --run-label "${OAMB_RUN_LABEL}-bounded-hindsight" \
  --output-root "$OAMB_WORK_DIR/bounded" \
  --result-map "$OAMB_WORK_DIR/results/bounded-hindsight.json"

uv run --locked oamb run "$PLAN" \
  --cell mem0-lme60 \
  --question 72e3ee87 \
  --run-label "${OAMB_RUN_LABEL}-bounded-mem0" \
  --output-root "$OAMB_WORK_DIR/bounded" \
  --result-map "$OAMB_WORK_DIR/results/bounded-mem0.json"

uv run --locked oamb run "$PLAN" \
  --cell openviking-lme60 \
  --question 72e3ee87 \
  --run-label "${OAMB_RUN_LABEL}-bounded-openviking" \
  --output-root "$OAMB_WORK_DIR/bounded" \
  --result-map "$OAMB_WORK_DIR/results/bounded-openviking.json"
```

Read the capsule paths from the result maps and validate their actual bytes:

```bash
HINDSIGHT_BOUNDED_ROOT="$(jq -er '.capsule_roots["hindsight-lme60"]' \
  "$OAMB_WORK_DIR/results/bounded-hindsight.json")"
MEM0_BOUNDED_ROOT="$(jq -er '.capsule_roots["mem0-lme60"]' \
  "$OAMB_WORK_DIR/results/bounded-mem0.json")"
OPENVIKING_BOUNDED_ROOT="$(jq -er '.capsule_roots["openviking-lme60"]' \
  "$OAMB_WORK_DIR/results/bounded-openviking.json")"

uv run --locked oamb capsule validate "$HINDSIGHT_BOUNDED_ROOT" \
  --output "$OAMB_WORK_DIR/validations/bounded-hindsight.json"
uv run --locked oamb capsule validate "$MEM0_BOUNDED_ROOT" \
  --output "$OAMB_WORK_DIR/validations/bounded-mem0.json"
uv run --locked oamb capsule validate "$OPENVIKING_BOUNDED_ROOT" \
  --output "$OAMB_WORK_DIR/validations/bounded-openviking.json"
```

Do not start LME-60 unless all three validation commands print `validated`.

### 5. Run and validate LME-60

The full command admits all three provider cells concurrently. It refuses to
start unless the three bounded capsules are fresh, valid, use the same frozen
question, and match the current provider project and resolved plan.

```bash
uv run --locked oamb run "$PLAN" \
  --run-label "${OAMB_RUN_LABEL}-full" \
  --output-root "$OAMB_WORK_DIR/full" \
  --result-map "$OAMB_WORK_DIR/results/full.json" \
  --bounded-capsule "hindsight-lme60=$HINDSIGHT_BOUNDED_ROOT" \
  --bounded-capsule "mem0-lme60=$MEM0_BOUNDED_ROOT" \
  --bounded-capsule "openviking-lme60=$OPENVIKING_BOUNDED_ROOT" \
  --bounded-validation \
    "hindsight-lme60=$OAMB_WORK_DIR/validations/bounded-hindsight.json" \
  --bounded-validation \
    "mem0-lme60=$OAMB_WORK_DIR/validations/bounded-mem0.json" \
  --bounded-validation \
    "openviking-lme60=$OAMB_WORK_DIR/validations/bounded-openviking.json"

HINDSIGHT_ROOT="$(jq -er '.capsule_roots["hindsight-lme60"]' \
  "$OAMB_WORK_DIR/results/full.json")"
MEM0_ROOT="$(jq -er '.capsule_roots["mem0-lme60"]' \
  "$OAMB_WORK_DIR/results/full.json")"
OPENVIKING_ROOT="$(jq -er '.capsule_roots["openviking-lme60"]' \
  "$OAMB_WORK_DIR/results/full.json")"

uv run --locked oamb capsule validate "$HINDSIGHT_ROOT" \
  --output "$OAMB_WORK_DIR/validations/full-hindsight.json"
uv run --locked oamb capsule validate "$MEM0_ROOT" \
  --output "$OAMB_WORK_DIR/validations/full-mem0.json"
uv run --locked oamb capsule validate "$OPENVIKING_ROOT" \
  --output "$OAMB_WORK_DIR/validations/full-openviking.json"
```

Each full validation must print `validated` before comparison.

### 6. Build and open the comparison report

```bash
uv run --locked oamb compare "$PLAN" \
  --cell-root "hindsight-lme60=$HINDSIGHT_ROOT" \
  --cell-root "mem0-lme60=$MEM0_ROOT" \
  --cell-root "openviking-lme60=$OPENVIKING_ROOT" \
  --validation \
    "hindsight-lme60=$OAMB_WORK_DIR/validations/full-hindsight.json" \
  --validation \
    "mem0-lme60=$OAMB_WORK_DIR/validations/full-mem0.json" \
  --validation \
    "openviking-lme60=$OAMB_WORK_DIR/validations/full-openviking.json" \
  --dataset-source \
    datasets/longmemeval-cleaned/longmemeval_s_cleaned.json \
  --output-root "$OAMB_WORK_DIR/comparison"

jq -e '
  .coverage == {
    "cell_count": 3,
    "unique_case_count": 60,
    "provider_specific_result_count": 180
  } and
  ([.cells[].completed_case_count] | all(. == 60)) and
  ([.cells[].accuracy.all_60.denominator] | all(. == 60)) and
  (.comparisons | length) == 3
' "$OAMB_WORK_DIR/comparison/report.json" >/dev/null

open "$OAMB_WORK_DIR/comparison/report.html"       # macOS
# xdg-open "$OAMB_WORK_DIR/comparison/report.html" # Linux
```

`report.html` is self-contained and makes no network requests. It presents
overall and per-question-type accuracy with 95% Wilson intervals, pairwise
exact McNemar evidence, context tokens, indexing tokens, latency, retries,
failures, resource and cost evidence, and measurement coverage. A provider is
shown as an observed accuracy leader only when it satisfies the frozen accuracy
delta and McNemar thresholds against both other providers.

### If a run stops or fails

Preserve `.local-demo`, `provider-services/.runtime`, and all provider state.
Do not delete or overwrite them. A failed model-readiness attempt cannot be
reused with a different project name because its runtime is already bound to
the old project. First stop that project's containers while preserving their
data:

```bash
./provider-services/bin/provider-services stop
```

Then start again in a separate fresh clone, which gives the new project its own
`provider-services/.runtime`. Choose a new `OAMB_PROVIDER_PROJECT`, run label,
and working directory there, and follow this Quick Start from the beginning.
Do not copy the failed clone's `.runtime` into the new clone. The stopped
project's Docker volumes and the original evidence remain preserved.

If `oamb run` fails after dispatch, inspect the `--result-map` path from that
command before doing anything else. It records each cell as completed, failed,
or not started and preserves every completed capsule path. If that map itself
could not be created, the command prints the same cell outcomes and paths with
both the run error and map error.

The 900-second operation timeout limits local waiting for one attempt. It does
not prove that remote work or remote billing stopped, so retain the evidence and
reconcile uncertain calls before starting another project.

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
