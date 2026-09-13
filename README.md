# Open Agent Memory Benchmark

## Overview

Open Agent Memory Benchmark (OAMB) compares Hindsight, Mem0, and OpenViking through their native REST APIs under one shared evaluation protocol. The v0.1.0 profile uses the same balanced selection of 60 LongMemEval questions for each provider: ten questions from each of six question types, producing 180 provider-specific results.

The offline report presents five metrics separately:

- **Answer accuracy** — correctness using retrieved memory.
- **Context tokens** — retrieved evidence shown to the answer model.
- **Indexing tokens** — supplier-reported generative producer tokens for building memory; embedding and provider user-provisioning setup are excluded.
- **Retrieval latency** — time spent on the provider's memory-query request.
- **Indexing time** — time from the first history write to query readiness.

OAMB fixes questions, model roles, answer/judge policy, and comparison settings before execution. Providers retain their native storage and indexing behavior. Retries, failures, partial ingestion, costs, and unavailable measurements remain visible; there is no universal combined score. Retrieval permits query embedding but disables generative query rewriting, reflection, and reranking.

## Provider versions

OAMB currently pins these formal provider releases and immutable source commits:

| Provider | Release | Source commit |
|---|---|---|
| Hindsight | [0.9.2](https://github.com/vectorize-io/hindsight/releases/tag/v0.9.2) | [`ebad478240d3171bb88201ececda5e8d9883d22d`](https://github.com/vectorize-io/hindsight/commit/ebad478240d3171bb88201ececda5e8d9883d22d) |
| Mem0 | [2.0.19](https://github.com/mem0ai/mem0/releases/tag/v2.0.19) | [`dc82354e143c2581d505d581a00286d6ef8c3605`](https://github.com/mem0ai/mem0/commit/dc82354e143c2581d505d581a00286d6ef8c3605) |
| OpenViking | [0.4.19](https://github.com/volcengine/OpenViking/releases/tag/v0.4.19) | [`f3afef11637f2d7c11e4b1f36ed2f90630737cdc`](https://github.com/volcengine/OpenViking/commit/f3afef11637f2d7c11e4b1f36ed2f90630737cdc) |

[`provider-services/versions.env`](provider-services/versions.env) is the authoritative runtime pin; image digests and source archive hashes there make provider preparation reproducible.

## Quick start

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
