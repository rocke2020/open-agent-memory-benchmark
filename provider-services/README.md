# OAMB provider API services

This bundle prepares exact-pinned local REST services for the Open Agent Memory
Benchmark (OAMB). It does not run LongMemEval, prove provider workload support,
or prove generation-free retrieval; those require the benchmark runner, runtime
evidence, and capsule validation.

The default transport-matched boundary calls Hindsight, Mem0, and OpenViking
through local REST APIs. Mem0's Python SDK is an optional, separately identified
profile; it is not part of this Compose project and its evidence must never be
mixed with REST evidence.

This bundle prepares exact-pinned services; it does not establish benchmark
profile support. v0.1 targets the same balanced LME-60 LongMemEval manifest for
all three providers. OpenViking requires a separately verified
session/message/commit adapter, while MemoryAgentBench remains deferred.

## Fixed releases

**All service identities are immutable and fail closed before reuse; a version
label alone is insufficient.**

| Provider | Release | Immutable runtime |
|---|---|---|
| Hindsight | `v0.9.2` | official API-only slim image by multi-arch digest |
| Mem0 | `v2.0.19` | exact source commit, archive hash, hash-locked dependencies, pinned Python base, and complete build-input fingerprint |
| OpenViking | `v0.4.16` | official image by multi-arch digest |

The exact commits and digests are in `versions.env`. The Mem0 build accepts
only an official local checkout whose `v2.0.19` tag resolves to the recorded
commit. Dirty worktree content and the checkout's current branch are excluded
because the build input is a digest-checked `git archive` of that commit.
The build-input fingerprint additionally binds the Dockerfile, dependency lock, inspector, extraction-response normalization, Docker ignore rules, and fixed source archive before an existing local image may be reused.

The Mem0 image includes a small [extraction-response normalization](mem0/extraction_response.py) before native embedding and storage in both synchronous and asynchronous ingestion. A decoded `memory` value may contain text strings or text-bearing objects, individually or in an array; the normalizer wraps strings as `text` objects and preserves existing object fields and text bytes. Empty arrays remain valid empty extraction; invalid or blank entries reject the entire batch before storage. This prevents the pinned parser's `.get()` crash without changing extraction prompts, models, attribution, or retry policy.

## State and safety boundary

Each `OAMB_PROVIDER_PROJECT` owns new Compose volumes. The project never mounts
an existing `~/.openviking`, Mem0 history directory, provider checkout, or
database. Only four HTTP API ports bind to `127.0.0.1`; PostgreSQL remains
private. `stop` preserves containers, volumes, and data and refuses to
run or memory-conformance lifecycle while `.runtime/active-operation` exists.

The commands in this directory do not run benchmark memory ingestion,
retrieval, LongMemEval, or deferred MAB research workloads. `verify --services`
may provision Mem0 configuration and an OpenViking
account/key, then proves restart persistence, non-ROOT auth, and OpenViking's
storage layout with a network-disabled read-only one-shot probe. Memory
conformance and paid model evaluation remain separate benchmark work.

## Prepare

**Preparation fails closed before service startup and keeps every real secret
in the local protected environment file.**

Requirements: Docker Engine with Compose, Git, Python 3, `curl`, `jq`, and
`shasum`. vLLM-metal must expose `qwen3-embedding:0.6b` on the configured
OpenAI-compatible endpoint, accept the native `dimensions: 1024` request, and
support at least an 8,192-token model and scheduler batch limit. Real LLM/VLM
credentials are needed only for the model-readiness mode and later memory evaluation. A
service-only acceptance may use clearly named nonfunctional values and a
`.invalid` base URL; such a run must remain `model_readiness=NOT_RUN`. Never
reuse another provider's tracked or local config.

```bash
cp .env.example .env
chmod 600 .env
# Edit the root .env and replace every change-me value.

uv run --locked oamb doctor configs/benchmark.yml --output outputs/tmp/provider-plan
. ./provider-services/lib/plan_environment.sh
load_plan_model_environment outputs/tmp/provider-plan/resolved-plan.json

./provider-services/bin/provider-services doctor
./provider-services/bin/provider-services build
./provider-services/bin/provider-services up
./provider-services/bin/provider-services verify --services
./provider-services/bin/provider-services verify --model-readiness \
  --resolved-plan /path/to/resolved-plan.json \
  --model-env .env
./provider-services/bin/provider-services status
```

The bundle accepts a deliberately small dotenv grammar: one unique
`NAME=value` per line, with no quoting, interpolation, backticks, backslashes,
or inline comments. This prevents the operator command and Compose from
resolving different values. Use URL-safe/hex-generated local secrets.

`doctor` is fail-closed: it validates tool availability, `.env` permissions,
required values, unique project naming, fixed Mem0 tag resolution, free ports,
and fully resolved Compose configuration without printing secrets. `build`
checks the source archive digest, builds Mem0 without a startup reinstall,
asserts its installed package version and OCI labels, and obtains the other
images by digest when they are not already present locally. Existing images
are reused only after their immutable digest and release labels pass. A local
Mem0 image is reused only when its full build-input fingerprint also matches.

`up` reports only container liveness. `verify --services` additionally proves
exact API versions/routes, storage reachability, Mem0 configuration persistence
across a normal restart, and an OpenViking user-bound key without benchmark-
memory writes. `verify --model-readiness` is a potentially billable endpoint-
readiness probe: it checks the common 1,024-dimension embedding response, a
minimal chat response from all three provider producer roles plus the harness
answer and judge roles, and OpenViking `/ready`. Service verification also
proves that provider-internal model/task retries are zero. It does not execute a
provider-native extraction path, validate structured extracted memory, or prove
the absence of a fallback model. A separate bounded fresh-scope question run is
therefore required for each provider before broad benchmark dispatch. v0.1 has
no retrieval-generation role. Neither probe may run concurrently with a
benchmark.

Model-readiness billing is explicitly `unavailable`, never zero. The command
loads the immutable resolved plan, uses its per-operation timeout, and probes
the three provider producer roles, controlled embedding, harness answer,
harness judge, and OpenViking readiness once each. Before the first dispatch it
creates one project-bound attempt and records each dispatch before sending it.
A failure cannot reuse the project; the operator preserves the attempt ledger
and starts a new project. The final receipt keeps `billing_complete=false` and
`cost_usd=null`.

The runner and lifecycle commands share one atomic local protocol. Before
creating `.runtime/active-operation`, a runner or conformance owner must acquire
`.runtime/provider-lifecycle.lock`, verify no lifecycle operation owns it,
create its lease, and release the lifecycle lock. `up`, `stop`, and both
`verify` modes hold the same lifecycle lock and refuse an active run. The
project attestation binds its name plus Compose/version hashes before any
mutation.

Service attestation, model readiness, and lifecycle ownership do not prove the
LongMemEval adapter or zero retrieval-generation dispatch; those require the
resolved cell, runtime outbound trace, and fresh capsule validation.

## Controlled embedding, native storage

All three default profiles use the same controlled embedding endpoint, model,
and dimension: `qwen3-embedding:0.6b` with 1,024 dimensions. They intentionally
do not share one vector database. Hindsight keeps its embedded pg0/pgvector
store, Mem0 uses its private PostgreSQL/pgvector store, and OpenViking uses its
embedded AGFS and local vector workspace. Storage and indexing are part of each memory
provider's native behavior; forcing one common database would bypass that
behavior and would no longer be the default provider comparison. A future
common-vector-store experiment must be a separately named ablation and may run
only when all selected releases officially support the same backend.

Query embedding is allowed non-generative retrieval infrastructure and remains
separate from prohibited generation paths. The v0.1 profiles select Hindsight
`recall` without `reflect`, Mem0 `/search` with effective native reranking
proven disabled, and OpenViking `/api/v1/search/find` with
`enable_intent=false`. Provider service health or source defaults alone cannot
prove these runtime settings.

OpenViking `/ready` performs a real embedding request and is intentionally not
a periodic healthcheck; model-readiness verification invokes it only under the
resolved plan's per-operation timeout and records that boundary separately from
service health.

## API endpoints

**Only provider REST APIs and the narrow read-only Mem0 projection inspector
are exposed on the host.**

Default host endpoints are:

- Hindsight: `http://127.0.0.1:18888`
- Mem0: `http://127.0.0.1:18889`
- Mem0 projection inspector: `http://127.0.0.1:16333`
- OpenViking: `http://127.0.0.1:19330`

The inspector supports authenticated `GET /health` and
`GET /v1/projection?run_id=<64-lowercase-hex>[&cursor=<opaque>]` only. It fixes
the `public.oamb_memories` table, binds the run ID and UUID cursor itself,
selects only IDs and JSON payloads, and never accepts arbitrary SQL, table names,
or write operations. The runner gets the inspector key, never the PostgreSQL
credential.

## Offline verification

**Credential-free contract checks validate the service tooling without making
provider or database writes.**

The contract suite is credential-free and performs no provider or database
writes:

```bash
python3 -m unittest discover -s provider-services/tests -p 'test_*.py' -v
docker compose --env-file /path/to/canary.env \
  -f provider-services/compose.yaml config --quiet
```

The second command still requires a mode-`0600` canary env containing all
required names. Never print resolved Compose output when real secrets are in
use.
