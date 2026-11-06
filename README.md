# Open Agent Memory Benchmark

## Overview

Open Agent Memory Benchmark (OAMB) focuses on evaluating Open Agent Memory (OAM), with Hindsight, Mem0, and OpenViking as comparison systems.

The saved OAM v0.0.1 development evaluations cover [LongMemEval's 500 questions](docs/evaluations/oam-v0.0.1-lme500.md) and [BEAM-10M's 10 conversations and 200 questions](docs/evaluations/oam-v0.0.1-beam10m.md). LME-500 achieved **94.0% (470/500)**; BEAM-10M's policy-final mean rubric score was **61.56%**, with **142/200** questions meeting the 0.5 threshold. These reports reuse completed campaigns and require no setup or new evaluation calls.

**We use only `deepseek-v4.1-flash` for extraction, answering and judging, prioritizing everyday agent tasks over peak leaderboard scores. Model choice matters:** Hindsight [reports **64.1%** on BEAM-10M](https://vectorize.io/benchmarks) using **Gemini 3.1 Pro** (`gemini-3.1-pro-preview`) for answering; our [DeepSeek replay of its saved contexts scored **52.16%** after 43 eligible max-effort answer retries](docs/evaluations/oam-v0.0.1-beam10m.md), a gap of **about 12 percentage points** in mean rubric score.

The existing executable comparison profile uses the native REST APIs of Hindsight, Mem0, and OpenViking on the same balanced selection of 60 LongMemEval questions: ten questions from each of six question types, producing 180 provider-specific results. The saved OAM evaluations remain separate historical evidence. The [native OAM workflows](docs/native-oam.md) now provide portable LME-500 and BEAM-10M execution through `./precheck.sh --oam` and `./run.sh --oam`, with small synthetic end-to-end diagnostics. Neither saved scores nor diagnostic passes certify the final release artifact.

The offline report presents five metrics separately:

- **Answer accuracy** — correctness using retrieved memory.
- **Context tokens** — retrieved evidence shown to the answer model.
- **Indexing tokens** — supplier-reported generative producer tokens for building memory; embedding and provider user-provisioning setup are excluded.
- **Retrieval latency** — time spent on the provider's memory-query request.
- **Indexing time** — time from the first history write to query readiness.

OAMB fixes questions, model roles, answer/judge policy, and comparison settings before execution. Providers retain their native storage and indexing behavior. Retries, failures, partial ingestion, costs, and unavailable measurements remain visible; there is no universal combined score. Retrieval permits query embedding but disables generative query rewriting, reflection, and reranking.

## Provider versions

OAMB targets **OAM v0.0.1** first, alongside these comparison-provider releases and source commits:

| Provider | Release | Source commit |
|---|---|---|
| Open Agent Memory (OAM) | **0.0.1 development snapshot** | `d41fd439956a7ec4e92d0d67185250be4e86ee35` |
| Hindsight | [0.9.2](https://github.com/vectorize-io/hindsight/releases/tag/v0.9.2) | [`ebad478240d3171bb88201ececda5e8d9883d22d`](https://github.com/vectorize-io/hindsight/commit/ebad478240d3171bb88201ececda5e8d9883d22d) |
| Mem0 | [2.0.19](https://github.com/mem0ai/mem0/releases/tag/v2.0.19) | [`dc82354e143c2581d505d581a00286d6ef8c3605`](https://github.com/mem0ai/mem0/commit/dc82354e143c2581d505d581a00286d6ef8c3605) |
| OpenViking | [0.4.19](https://github.com/volcengine/OpenViking/releases/tag/v0.4.19) | [`f3afef11637f2d7c11e4b1f36ed2f90630737cdc`](https://github.com/volcengine/OpenViking/commit/f3afef11637f2d7c11e4b1f36ed2f90630737cdc) |

[`provider-services/versions.env`](provider-services/versions.env) pins the comparison-provider images and source archives. The OAM development image is built from the source commit above using the instructions below.

### OAM development image

Build the OAM v0.0.1 development image from a checkout containing the recorded commit:

```bash
./provider-services/oam/build-image.sh /path/to/open-agent-memory d41fd439956a7ec4e92d0d67185250be4e86ee35
docker run --rm oamb/oam:0.0.1-d41fd439956a --version
```

The build uses committed source and the OAM lockfile, excluding local uncommitted files. The resulting local image is a development snapshot, not the published release image. A working OAM deployment still requires an isolated PostgreSQL database, appropriate credentials and configured model services. Use the [native OAM setup and execution guide](docs/native-oam.md) for isolated PostgreSQL/API/worker preparation, input and model binding checks, token ceilings, checkpoints and resume.

The evaluation overrides in `provider-services/oam/` configure LME-500 for the shared `small` vector index and BEAM-10M for `large` per-user indexes. Both disable Insights and Profile and select low extraction reasoning. The evaluator must request fast recall for LME and normal recall for BEAM; these modes are request parameters, not index settings. Other settings come from the image's canonical OAM configuration and must be checked against the historical protocol before claiming a matched rerun.

The historical evaluations used Qwen3 embedding with a 32K-token input limit and OAM's approximate tokenizer. Current OAM defaults ingestion packing to `o200k_base` using `toktok-rs`, so the supplied image and overrides do not recreate the historical tokenizer behavior by themselves. For more accurate chunk sizing, prefer the embedding model's own tokenizer and configure its actual input and request limits. For example, `cl100k_base` matches OpenAI's `text-embedding-3` models. Check the exact model's tokenizer, input formatting and limits before use; the default encoding is not a universal match.

Current OAM supports `o200k_base`, `cl100k_base`, the legacy `oam-unicode-quarter-v1` counter and a model-specific `tokenizer_json` asset. Set `tokenizer.type` for ingestion packing and, when needed, a separate `models.embedding.tokenizer` for embedding admission. Configure `models.embedding.max_input_tokens`, `max_request_tokens` and `max_request_items` to match the service. If no supported counter matches the model, treat the count as an approximation and allow a safety margin; it cannot guarantee that input fits the provider's limit. Verify the embedding dimension as well. Changing the embedding model or tokenizer may change chunking, embeddings and results, so report it rather than presenting the rerun as identical to the historical campaign.

The native BEAM workflow provisions a dedicated index after each conversation's embedded search units exist. Both profiles verify the expected index against the actual dense-discovery query plan before scoring. Small synthetic samples have exercised real ingestion, recall, answering, judging and offline HTML inspection; they are separate from historical and full-dataset results. See the [workflow guide](docs/native-oam.md) for exact commands, artifact boundaries and current limitations.

## Quick start

The commands below start a separate native provider comparison. They are unnecessary for reading the saved [LME-500](docs/evaluations/oam-v0.0.1-lme500.md) and [BEAM-10M](docs/evaluations/oam-v0.0.1-beam10m.md) evaluations.

You need Git, Python 3.11+, `uv`, Docker with Compose, `curl`, `jq`, `shasum`, and credentials for an OpenAI-compatible model endpoint. Preparation and evaluation can make billable model calls. They create isolated provider state and preserve existing provider/database data.

### 1. Configure

Clone the repository and prepare its private configuration:

```bash
git clone https://github.com/rocke2020/open-agent-memory-benchmark.git
cd open-agent-memory-benchmark
cp .env.example .env
chmod 600 .env
```

Edit `.env`: set `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_LIGHT_MODEL`, and `LLM_DEEP_MODEL`; keep `LLM_URL_TYPE=openai_chat`. The [template](.env.example) uses `deepseek-flash` for both model profiles. The light profile handles provider extraction and judging; the deep profile handles answering. [Benchmark configuration](configs/benchmark.yml) owns role assignments, thinking effort, and execution controls.

To reuse an embedding service, set its OpenAI-compatible base URL. The default profile requires `qwen3-embedding:0.6b`, 1,024-dimensional vectors, and sufficient context capacity for the providers' source-session inputs. For an existing host-local service:

```dotenv
OAMB_EMBEDDING_BASE_URL=http://127.0.0.1:18000/v1
OAMB_EMBEDDING_API_KEY=
```

The embedding API key may be missing or empty for a keyless service. OAMB translates this HTTP IPv4 loopback URL for Docker containers while keeping the host probe unchanged. Remote HTTPS endpoints are also supported; HTTPS and IPv6 loopback URLs are rejected.

If the URL is unset or remains `change-me`, precheck uses the local fallback: macOS requires a sibling `vllm-metal` installation and cached model files; Linux requires Ollama and may download the embedding model on first use. See the [local embedding helpers](scripts/start_local_embedding).

### 2. Precheck and run

Prepare dependencies, the dataset, pinned provider services, and model readiness:

```bash
./precheck.sh
```

Continue only after `precheck: PASS`. Then run the complete comparison:

```bash
./run.sh --full_test
```

Providers run concurrently within the configured limits. Each provider prints its status, elapsed time, and saved-question count every 30 seconds. Completed results are saved atomically; the final comparison requires all 180 results and fresh evidence validation.

Other run modes:

```bash
./run.sh --full_test --resume  # Reuse completed questions; rerun missing ones in fresh scopes
./run.sh --smoke_test         # Optional one-question diagnostic across all three providers
./run.sh --dry-run            # Check the frozen plan without model or provider calls
```

Plain `./run.sh` also selects smoke mode. Smoke results are separate and are not required for a full run.

Set concurrency under `execution` in [configs/benchmark.yml](configs/benchmark.yml):

| Setting | Limits |
|---|---|
| `max_parallel_providers_per_dataset` | Concurrent provider evaluations |
| `max_parallel_history_ingestions_per_provider` | Independent histories being ingested per provider |
| `max_parallel_questions_per_provider` | Retrieval, answer, and judge pipelines per provider |

Set the shared native candidate ceiling with `retrieval.top_k` in the same configuration. The value is frozen into the resolved plan and used by every provider adapter: Mem0 sends it as `top_k`, OpenViking sends it as `limit`, and Hindsight keeps only the first N provider-ranked normalized candidates because its recall endpoint has no item-count parameter.

Sessions within one history remain sequential. These limits do not cap a provider's internal embedding/model requests. Fresh runs use the prechecked plan. Each `--resume` reads the current YAML's history and question caps for unfinished work; changing those two values does not invalidate completed results. Other plan settings stay frozen, and already-running processes do not reload YAML. Do not edit `resolved-plan.json`.

Retry and per-attempt timeout settings are documented alongside these controls in the configuration. They are not whole-run or spending caps. A local timeout does not prove remote processing or billing stopped.

### 3. Stop provider services

Press `Ctrl-C` once to stop an active full run and its provider containers. To stop this repository's provider-service containers directly:

```bash
./provider-services/bin/provider-services stop
```

Stopping preserves Docker volumes, provider/database data, saved results, logs, and evidence. It clears ephemeral lifecycle markers after provider execution ends. Use the resume command above to continue an interrupted full run.

## Results and troubleshooting

Full reports are written under `outputs/full-test/<run-label>/`; smoke reports use `outputs/smoke-test/<run-label>/`. The self-contained HTML opens automatically. Set `OAMB_NO_OPEN=1` for headless use; the report is still generated and its path printed.

### Generate a report from separate provider results

When Hindsight, Mem0, and OpenViking were completed and preserved as separate snapshots beneath one directory, generate the report without rerunning providers:

```bash
./run.sh --generate-report --result-dir=outputs/saved-results/lme60-provider-native
```

The directory must contain exactly one complete 60-question snapshot for each provider, with `resolved-plan.json` and `results/<provider>.json` in each snapshot. The command requires the root `.env` for the frozen judge-model connection, makes one bounded LLM analysis call when no matching analysis is cached, and reuses the content-addressed cache at `<result-dir>/report-analysis-cache` without another model call when the report inputs and analysis configuration are unchanged. It requires no precheck or provider service, leaves the source snapshots unchanged, and writes a create-only `comparison/` directory (or a timestamped successor when that name already exists). It records every source plan/result hash and suppresses an affected leader claim when the saved plans do not prove the same comparison control.

The HTML keeps the comparison focused: it omits unavailable secondary accounting and Wilson-interval text, and its Question results section shows only questions answered incorrectly by at least one provider. The complete 60-question matrix and statistical evidence remain in `report.json`.

### Check saved accuracy

Answer accuracy is one of the primary metrics. For a complete or interrupted full run, use the saved result map to calculate `correct / judged`; report `saved / 60` separately as completion coverage.

```bash
result_file=".../$provider.json"
jq -r '
  [.[] | select(.evaluation.disposition == "judged")] as $judged
  | ($judged | map(.evaluation.numerator) | add // 0) as $correct
  | ($judged | map(.evaluation.denominator) | add // 0) as $total
  | if $total == 0 then "accuracy: unavailable; completion: \(length)/60"
    else "accuracy: \($correct)/\($total) = \(((10000 * $correct / $total | round) / 100))%; completion: \(length)/60"
    end
' "$result_file"
```

Every run prints `run: log=<path>` and saves terminal output under `outputs/tmp/`. Follow it with `tail -f <path>`. Full-run progress comes from saved `results/{hindsight,mem0,openviking}.json`; resume counts include earlier completed questions.

See the [Investigations](docs/investigations/README.md) for observed provider behavior and failures.

## Acknowledgments

Thank you to Vectorize's [Agent Memory Benchmark (AMB)](https://github.com/vectorize-io/agent-memory-benchmark) and its contributors for publishing their evaluation harness, methodology, prompts, and results. AMB was an important reference for OAMB's evaluation workflow and design. OAMB is independently implemented.

We also thank the [LongMemEval authors](https://github.com/xiaowu0162/LongMemEval) for the dataset and judge rubrics, and the [Hindsight](https://github.com/vectorize-io/hindsight), [Mem0](https://github.com/mem0ai/mem0), and [OpenViking](https://github.com/volcengine/OpenViking) communities for their open-source memory systems. Directly reused prompt materials retain their [source attribution and notices](prompt-packs/README.md).

## Project policies

See [Security](SECURITY.md), [Dataset provenance](DATASETS.md), and [Third-party components](THIRD_PARTY.md).

OAMB is licensed under [Apache-2.0](LICENSE). Datasets and third-party artifacts retain their own licenses.
