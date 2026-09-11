# OpenViking and Mem0 ingestion latency under shared local embeddings

## Overview

The September 11, 2026 investigation found an OpenViking-specific batch wait that repeats during sequential history ingestion, additional LLM extraction work, and different embedding request patterns between OpenViking and Mem0. Sharing one embedding service therefore does not give the two providers equivalent ingestion costs.

Mem0 finishing coincided with a substantial reduction in embedding workload and the operator's observation that the Mac became quieter and cooler. OpenViking's median session-ingestion latency barely changed in the short interval afterward. The evidence supports Mem0 contributing substantially to machine load, while OpenViking retains its own ingestion delays. It does not establish a defective vLLM-Metal configuration or a precise allocation of total latency among causes.

This is a dated operational investigation, not a controlled provider performance ranking or a final evaluation report. The investigation used existing artifacts, logs, source inspection, and read-only runtime checks. It did not modify services, configuration, benchmark results, or provider data, and did not submit additional model requests.

## 1. Scenario and measurement boundaries

The question was why OpenViking ingested histories more slowly than Mem0 when both used the same local embedding service, and whether Mem0's concurrent work explained the Mac's heat and fan noise.

- **History:** all source conversations supplied for one benchmark question. Histories are isolated from one another.
- **Session:** one source conversation inside a history. Sessions within a history are ingested sequentially.
- **Commit:** OpenViking's operation that extracts memories from a session and waits for its native processing to finish.
- **Peer:** the OpenViking identity used to isolate a benchmark history. Different concurrent histories use different peers.
- **Embedding throughput:** input tokens processed per second, as reported by vLLM. This measures processed workload, not electrical power or GPU utilization.
- **P95:** the 95th-percentile observation. The diagnostic calculation uses the sorted sample at zero-based index `floor(0.95 * (n - 1))`.

The inspected environment was an Apple M4 Pro Mac mini with 64 GB unified memory and an internal SSD. Mem0 v2.0.19 and OpenViking v0.4.16 ran in Docker. Both used a host vLLM-Metal service running Qwen3-Embedding-0.6B Q8_0, pooling mode, and 1,024-dimensional output; the installed vLLM version was 0.27.0. OpenViking used its `v2` memory profile, with session skill extraction disabled. Both providers used `deepseek-flash` for extraction-related generation.

The active resumed run retained six history-ingestion workers and six question workers per provider from its saved plan. The operator's subsequent YAML edit to `3/3` did not change those running limits. History-worker counts do not equal concurrent embedding-request counts: providers have their own batching and internal concurrency.

Times below are UTC unless otherwise stated; local time was UTC+8. Measurements come from one resumed invocation that began around `15:12:18`. Completed-question counts include reused results, so their raw ratio is not a speed comparison. Session-ingestion durations below use completed `memory_ingest` attempt timestamps from that invocation.

## 2. Observed ingestion difference

OpenViking's completed session-ingestion operations were substantially slower and had a longer tail in the inspected invocation. The providers were processing different remaining histories, so these measurements diagnose the running workload rather than establish an equal-input performance ratio.

Snapshot at approximately `15:44:43`:

| Provider | Completed session ingestions | Median | Mean | P95 | Maximum |
|---|---:|---:|---:|---:|---:|
| Mem0 | 807 | 14.07 s | 14.34 s | 23.10 s | 37.69 s |
| OpenViking | 294 | 30.47 s | 39.38 s | 98.24 s | 274.58 s |

These are whole ingestion-attempt durations, including the adapter's required work, rather than embedding-only latency. A preceding sample of 254 unique completed OpenViking tasks had a native commit median of `30.38 s`, close to the adapter-level median. This locates substantial time inside native commit processing. OAMB polls for completion every `0.1 s`; that configured polling interval does not explain a repeated ten-second wait.

At this snapshot, saved question counts were Mem0 `55/60` and OpenViking `16/60`. Later, Mem0 reached `60/60` while OpenViking was at `18/60`. These are dated partial-progress observations, not a claim about the final three-provider result.

## 3. Confirmed mechanism: OpenViking's small-update batch wait

OpenViking's native memory updater buffers ordinary updates until it has eight operations or a ten-second wait expires. Because the batch is separated by peer and memory type, OAMB's isolated, sequential histories often cannot fill it before the timer fires. This adds a repeated delay without indicating slow physical storage.

The pinned runtime's `StreamingMemoryUpdaterConfig` defaults were `max_operations_per_update=8`, `max_wait_seconds=10.0`, and `timer_check_interval_seconds=1.0`. A commit awaits completion of its submitted updates. Append-only event memories use an immediate path; ordinary preference, entity, and profile updates use the buffered path. [OpenViking updater source](https://github.com/volcengine/OpenViking/blob/v0.4.16/openviking/session/memory/streaming_memory_updater.py)

The interaction with the benchmark is direct: a session produces a few updates for a peer and memory type, its commit waits for those updates, and the next session in that same history cannot start until the commit finishes. Other histories have different peers and cannot contribute to that batch. Eight operations across an entire session do not suffice if its individual buffered groups contain fewer than eight operations.

Two completed sessions provide live supporting evidence:

| Sample | Native task duration | Immediate event write | Buffered write | Event-to-buffered-write gap |
|---|---:|---|---|---:|
| A | 23.929 s | `15:40:37.266` | Preference at `15:40:47.939` | 10.673 s |
| B | 29.907 s | Events at `15:40:39.098–39.103` | Entities at `15:40:49.720`; preference/profile at `15:40:49.978–49.980` | 10.62–10.88 s |

Within each sample, the immediate and delayed memory files shared the same `source_extraction_id`, tying them to the same extraction. Sample B contained four immediate event writes and buffered groups of two entities, one preference, and one profile update. Its eight aggregate operations still left each buffered group below the threshold.

The measured gaps are filesystem modification-time differences, not instrumented timer spans; they include application and filesystem overhead. Their agreement with the verified execution path supports the batching explanation. Detailed native tracing was disabled, so the exact timer-only duration and its contribution across all sessions were not measured.

Relevant benchmark behavior is in [the OpenViking session adapter](../../src/oamb/memory_systems/openviking/session_adapter.py), which waits for the native task before continuing the history. The inspected [OpenViking service configuration](../../provider-services/openviking/ov.conf) does not override the native batch-window defaults.

## 4. Additional cost: extraction and embedding request patterns

OpenViking's ingestion path contains more stages than Mem0's normal successful path, and the shared embedding endpoint receives different request patterns. These differences explain why server sharing alone does not imply similar completion times; the precise latency contribution of each stage remains unmeasured.

| Stage | Mem0's inspected path | OpenViking's inspected path |
|---|---|---|
| Extraction | Session embedding for existing-memory recall, followed by one successful extraction/merge LLM call; failed calls may retry | Archive summarization alongside an extraction loop that can make multiple LLM/tool rounds and repair operations |
| Memory application | Applies extracted operations without the corresponding ten-second native batch window | Immediate append-only writes plus buffered ordinary updates, awaited by commit |
| Indexing embeddings | Batches extracted texts, up to 100 per HTTP request | Queues changed memories individually; the inspected indexing path calls the embedder per item |
| Next source session | Starts after the current ingestion completes | Starts after the current commit and required adapter checks complete |

Mem0's embedding batch behavior is explicit in [its embedding implementation](https://github.com/mem0ai/mem0/blob/v2.0.19/mem0/embeddings/openai.py). OpenViking's extraction loop is implemented in [its extraction source](https://github.com/volcengine/OpenViking/blob/v0.4.16/openviking/session/memory/extract_loop.py). Application-level single-item requests can still be combined by vLLM's scheduler; they do not prove one GPU invocation per HTTP request.

### 4.1. Mem0 source trace: batching and concurrency are different

The deployed Mem0 source confirms application-level batching, with concurrency supplied by overlapping history requests. Its embedding helper does not create parallel workers for its own chunks. Fewer HTTP requests and bulk persistence are concrete efficiencies; the resulting GPU speedup, energy efficiency, and fraction of the provider latency gap were not measured in isolation.

OAMB explicitly sends `infer=true` through [the Mem0 request encoder](../../src/oamb/memory_systems/mem0/wire.py), selecting the ordinary extraction path. The deployed REST handler is synchronous and calls `Memory.add()`; FastAPI runs synchronous handlers in its worker pool. The separately available `AsyncMemory` implementation is not the entrypoint used by this REST profile.

| Order | Source in the deployed Mem0 package | Work performed |
|---|---|---|
| 1 | `memory/main.py`, `_add_to_vector_store()`, lines 919–932 | Embed the input session once to retrieve relevant existing memories |
| 2 | `memory/main.py`, lines 941–967 | Make one successful extraction LLM call; extraction failures may retry |
| 3 | `memory/main.py`, lines 974–986 | Collect nonempty extracted memory texts and pass the list to `embed_batch()` |
| 4 | `memory/main.py`, lines 988–1067 | Deduplicate records, insert vectors in bulk, and write history records in a batch |
| 5 | `memory/main.py`, lines 1069–1099 | Extract entity names locally, deduplicate them across these memories, and embed the unique names as another batch |

The source is [Mem0's ingestion implementation](https://github.com/mem0ai/mem0/blob/v2.0.19/mem0/memory/main.py). OAMB's deployed extraction retry adjustment does not change the batching sequence. Notably, memory-text hash deduplication happens after those texts are embedded; entity-name deduplication happens before entity embedding. The source does not justify claiming that all duplicate embedding work is eliminated.

In `embeddings/openai.py`, `embed_batch()` splits the text list into chunks of at most 100 and sends each chunk as one request's `input` array. Its synchronous loop waits for each response before sending the next chunk, restores vector order using response indices, and verifies that the returned vector count matches the input count. Thus 250 texts require three sequential HTTP requests within that call, rather than 250 single-text requests or three internally parallel requests.

For a successful ingestion with `M` extracted memory texts and `E` unique entity names, the normal path makes approximately `1 + ceil(M / 100) + ceil(E / 100)` embedding requests. Empty stages are skipped, and error fallback can add individual requests. For example, 20 memory texts and seven entity names normally require three embedding HTTP requests: the initial session embedding, one memory batch, and one entity batch.

Across six independent histories, those otherwise sequential pipelines can overlap. If six histories reach a 20-text embedding stage together, the server receives six HTTP requests containing 120 texts. That is a workload example, not an observed GPU batch: vLLM still determines execution batches subject to its scheduler and token limits. OpenViking also has concurrent embedding workers, so describing OV as entirely serial would be incorrect; its inspected indexing path uses concurrent single-item requests instead of Mem0's explicit multi-text request batches.

Mem0's bulk storage behavior is also explicit in [its pgvector implementation](https://github.com/mem0ai/mem0/blob/v2.0.19/mem0/vector_stores/pgvector.py): `insert()` passes the vector records through `executemany()` or `execute_values()`, depending on the installed PostgreSQL driver. This is an additional structural efficiency, not proof that storage explains the measured latency difference.

### 4.2. Native generation evidence

The long OpenViking commits contain extensive LLM generation, providing another reason to investigate the extraction path independently of embedding throughput.

One completed OpenViking native commit took `274.445 s` and reported `63,234` LLM reasoning tokens, `68,007` total completion tokens, and `2,662` embedding input tokens. This is evidence of extensive generation in a slow commit. It does not, by itself, give the fraction of wall time spent generating, because native stages can overlap and token categories represent different work.

The inspected log window beginning at `15:12:18` contained four malformed memory-operation output errors and four iteration extensions for patch repair. Those are distinct observations, not necessarily eight independent failed sessions. Successful native recovery can add work without causing a terminal ingestion failure. No change to extraction prompts, model behavior, or retry policy was made during this investigation.

## 5. Mem0 finishing, embedding load, and Mac cooling

The workload reduction when Mem0 finished supports the operator's cooling observation, but the short comparison does not show a material improvement in OpenViking's median session latency. Heat and fan noise, shared-resource contention, and provider-specific processing delay must be evaluated separately.

The last Mem0 ingestion ended at `15:46:29.399`, or `23:46:29.399` local time. The following comparison used existing logs and completed attempts:

| Measurement | Before final wind-down | After Mem0 ingestion finished |
|---|---:|---:|
| Observation window | `15:40:00–15:44:00` | Approximately `15:46:29–15:49:30` |
| vLLM throughput log samples | 24 | 19 |
| Mean reported embedding prompt throughput | 1,487.5 tokens/s | 345.4 tokens/s |
| Completed OV sessions wholly inside the window | 37 | 20 |
| OV session-ingestion median | 29.14 s | 28.71 s |
| OV session-ingestion mean | 32.55 s | 40.39 s |

Reported token throughput fell by approximately 77%. This is a reduction in processed embedding workload, not a measured 77% reduction in GPU utilization, power, or temperature. The operator independently reported that fan noise and perceived temperature decreased. No numerical temperature or fan-speed time series was collected, and `pmset -g therm` reported no recorded thermal or performance warning; that output is not a temperature reading or proof that throttling never occurred.

The windows differ in length and source sessions. Throughput log entries describe preceding intervals, so a boundary sample can straddle Mem0's final ingestion. The post-Mem0 sample is small and can contain outstanding shared-service work. Its nearly unchanged OV median and higher mean do not establish an improvement or regression, but they show that substantial OV latency remained after Mem0 stopped supplying ingestion work.

The supported interpretation is that six concurrent Mem0 histories contributed substantial demand to the shared local embedding service. Resource contention could have increased individual embedding latencies, and reduced demand is consistent with the quieter, cooler machine. The observations do not establish that six workers are intrinsically excessive for this hardware or that contention is the sole cause of OV's ingestion delay.

## 6. Metric interpretation and hypotheses not established

Zero vLLM running/waiting gauges cannot rule out contention in this pooling implementation. The investigation corrected an earlier interpretation that treated those values as proof of an idle or unqueued service.

In the inspected installed vLLM 0.27.0 source, queue time is calculated from scheduler admission to first scheduling. Metal pooling can execute synchronously while new requests wait before scheduler admission. Those requests have not yet entered the measured queue interval. Finished pooling requests can also be removed before the scheduler reports its running/waiting counts, producing zero gauges alongside nonzero throughput.

The difference between vLLM's end-to-end and inference timing can include tokenization, waiting before scheduler admission, interprocess communication, and frontend scheduling. It does not identify response-JSON serialization as the cause; HTTP response processing occurs after the measured frontend output interval. Relevant inspected paths are `vllm/v1/metrics/stats.py`, `vllm/v1/core/sched/scheduler.py`, `vllm/v1/engine/core.py`, `vllm_metal/v1/model_runner.py`, and `vllm_metal/v1/pooling.py` in the installed runtime.

The launcher used a `0.10` Metal memory fraction and `32768` model-length and batched-token limits. The memory fraction controls memory budgeting, not a 10% GPU-compute throttle. The token limits do not pad every request to 32,768 tokens. No controlled comparison established these settings as a performance regression. See the [embedding launcher](../../scripts/start_local_embedding/start_vllm_metal.sh).

The disk hypothesis also lacked supporting saturation evidence. The host used an SSD, OpenViking stored its workspace in a Docker volume, sampled host disk traffic was modest, and one ten-second container sample showed unchanged block-I/O counters while other activity continued. These brief observations do not prove storage has zero cost, but they do not support a slow mechanical disk as the explanation.

## 7. Evidence references and next experiments

The established first optimization target is the interaction between OV's small-update batch window and sequential history ingestion. Concurrency and model changes require separate experiments before claiming a speed or thermal benefit.

The investigation's raw evidence remains in the operator's local run artifacts and provider workspace; it is not included in this documentation directory. This report preserves the measured summaries and source references, but is not a standalone reproducible benchmark capsule.

| Evidence | Local record or inspection |
|---|---|
| Per-session ingestion durations | Active invocation's `source/attempts/*.json`, filtered to `stage=memory_ingest`; duration is `ended_at - started_at` |
| Saved question completion | The selected run's `results/{hindsight,mem0,openviking}.json` map lengths |
| Sample A task and memory timing | Native task `bdf0f13e-2b52-496b-8ad9-589d37db2a68`, its archive `memory_diff.json`, and referenced memory-file metadata |
| Sample B task and memory timing | Native task `b37bb479-ccf4-4bbc-877a-da860ff311a1`, its archive `memory_diff.json`, and referenced memory-file metadata |
| Long-generation outlier | Native task `ad757f6e-2986-4bbb-bbaa-b1b34d33ded9`, including its completed token-usage snapshot |
| Workload change | Timestamped vLLM prompt-throughput logs around Mem0's final ingestion |
| Native batching and request paths | Source installed in the pinned provider containers; separately installed vLLM/Metal source for metric semantics |

Potential follow-up experiments, not performed here:

1. Compare OV's existing batch window with a smaller window using the same representative histories and fresh isolated scopes. Verify memory outcomes as well as time; any non-default provider behavior must be disclosed in benchmark reporting. Do not change append-only/update semantics to bypass the wait.
2. Compare the same OV workload with and without concurrent Mem0 ingestion. Record embedding input sizes and latency distributions alongside native commit duration to isolate shared-service contention.
3. Test `3/3` against `6/6` in separate fresh runs. Lower concurrency reduces admitted work, but may increase total elapsed time or leave peak resource use unchanged. It does not remove OV's native batch wait, and changing YAML does not reconfigure an active resumed run.
4. Enable narrowly scoped native timing for an isolated diagnostic run if a precise split among extraction, batching, and embedding is needed. Treat temperature, fan speed, power, and throughput as separate measurements rather than inferring one from another.

No runtime fix or performance improvement was delivered as part of this investigation. The completed outcome is a source-grounded explanation of the repeated batch delay, a measured provider comparison with explicit limits, and evidence supporting the operator's observation about Mem0-related machine load.
