# Open Agent Memory Benchmark

> **TL;DR:** Open Agent Memory Benchmark (OAMB) keeps execution evidence,
> validation, metrics, resource usage, cost, and reports separate so a partial
> run cannot be mistaken for a score. The planned v0.1.0 comparison covers
> Hindsight, Mem0, and OpenViking on LongMemEval only. MemoryAgentBench is
> deferred. The OpenViking session profile and generation-free retrieval remain
> unverified until live preflight, execution, and capsule validation pass.

## Current status

**Implemented offline capabilities, planned v0.1 scope, and live-verified
provider support are separate states.** The repository contains exact-pinned
REST service preparation for Hindsight, Mem0, and OpenViking, a credential-free
fake path, fail-closed validation, reducers, deterministic offline reports, and
Linux/macOS clean-checkout CI. The current product refactor targets one generic
`doctor → run → validate → compare → report` flow; it is not complete merely
because the older fake path works.

The v0.1 execution target is three LongMemEval cells: Hindsight, Mem0, and an
OpenViking session/message/commit profile. MAB research artifacts do not
constitute v0.1 support, and no OpenViking LongMemEval claim is valid before its
new profile passes live gates.

No fixture, healthy service, non-empty response, or generated report implies a
live provider passed, a benchmark score is valid, or a release is ready.

## Credential-free implementation check

**A source checkout can exercise the existing fake evidence path from manifest
to offline report, but this is not the target v0.1 provider workflow.**
Requirements are Python 3.11 or newer and
[`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --locked --all-groups

mkdir -p .local-demo/inputs .local-demo/capsules .local-demo/reports

uv run oamb manifest build \
  --workload fake \
  --run-id readme-fake-run \
  --output .local-demo/inputs

uv run oamb preflight \
  --run-spec .local-demo/inputs/run-spec.json \
  --artifact-root .local-demo/capsules \
  --output .local-demo/resolved-plan.json

uv run oamb run --resolved-plan .local-demo/resolved-plan.json

uv run oamb capsule validate \
  .local-demo/capsules/readme-fake-run \
  --output .local-demo/evidence-validation.json

uv run oamb summarize \
  .local-demo/capsules/readme-fake-run \
  --validation .local-demo/evidence-validation.json \
  --output .local-demo/run-summary.json

uv run oamb report build \
  .local-demo/capsules/readme-fake-run \
  --validation .local-demo/evidence-validation.json \
  --output-root .local-demo/reports \
  --audience public
```

The commands are offline after dependencies are available. They use generated
fake components, make no provider calls, and write only beneath `.local-demo/`.

## External historical evidence

**The packaged historical view rebuilds from a restricted factual pack without
the raw producer result.** It remains `external_amb_generated`, compatibility is
`unknown`, and it contains no OAMB-native attempts, indexing usage, billing
completeness, comparison, or winner claim.

```bash
mkdir -p .local-demo/external .local-demo/external-report

uv run oamb external validate \
  --output .local-demo/external/validation.json

uv run oamb external report \
  --validation .local-demo/external/validation.json \
  --output-root .local-demo/external-report
```

`oamb external import --source ... --output ...` is a separate read-only
provenance command for the one pinned raw producer artifact. It first verifies
the complete source byte count and SHA-256, then emits only the repository-owned
field allowlist. It cannot import an arbitrary self-attested result.

## Runtime contracts

**The current refactor converges product artifacts on one strict, immutable
initial-v1 Pydantic family rather than checked-in generated schema files.**
Persisted models retain `schema_name` and `schema_version` wire discriminators;
semantic validators enforce identities, cross-artifact references, accounting
completeness, and report eligibility. Unreleased historical forms are not
public compatibility promises.

## Production workflow boundary

**Provider configuration, live execution, validation, and eligible comparisons
remain explicit operations, not installation claims.** Do not
translate the fake commands into a real provider run or reuse their artifact
root for live evidence. Every live call requires an explicit scope, immutable
runtime binding, bounded budget, unique artifact root, stop conditions, and retained
raw receipts.

## v0.1 retrieval boundary

**Retrieval generation is disabled.** Query embedding remains allowed, but
query planning, rewriting, decomposition, reflection, generative reranking,
and fallback to a generation-model retrieval path are prohibited. The selected
routes are Hindsight `recall` without `reflect`, Mem0 `/search` with effective
native reranking disabled, and OpenViking `/api/v1/search/find` with
`enable_intent=false`.

These are release-profile constraints, not claims about every provider mode.
Only complete runtime outbound evidence may label a finished cell
`runtime_verified`; source inspection or configuration defaults alone cannot
prove zero generative retrieval calls.

## Provider API services

**Service preparation is non-destructive and separate from benchmark
execution.** The default comparison boundary uses REST for Hindsight, Mem0,
and OpenViking. The optional Mem0 Python SDK profile is separate and cannot
substitute for REST evidence. See
[`provider-services/README.md`](provider-services/README.md) for the
non-destructive service-only workflow.

Start the services explicitly with:

```bash
cd provider-services && ./bin/provider-services up
```

The provider bundle prepares services only. v0.1 uses LongMemEval for all three
providers, and OpenViking requires the separate session/message/commit profile;
service health does not establish that workload support. Provider state is
preserved. Normal tests never delete databases or execute a paid evaluation.

## Project policies

**Security, dataset, and dependency policies are maintained as repository
documents.**

- [`SECURITY.md`](SECURITY.md) — vulnerability reporting and secret safety.
- [`DATASETS.md`](DATASETS.md) — dataset provenance and redistribution rules.
- [`THIRD_PARTY.md`](THIRD_PARTY.md) — direct dependency and license inventory.

Licensed under Apache-2.0. Dataset and third-party artifacts retain their own
licenses and are not relicensed by OAMB.
