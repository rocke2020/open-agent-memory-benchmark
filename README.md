# Open Agent Memory Benchmark

> **TL;DR:** Open Agent Memory Benchmark (OAMB) keeps execution evidence,
> validation, metrics, resource usage, cost, and reports separate so a partial
> run cannot be mistaken for a score. The repository offers a complete
> credential-free fake workflow, a hash-pinned external historical report, and
> approval-scoped phase-review commands. None implies that a live provider run
> passed.

## Current status

**The offline evidence pipeline and standalone distribution are implemented and
testable without credentials, while live benchmark execution remains gated.**
The repository contains
exact-pinned REST service preparation for Hindsight, Mem0, and OpenViking,
strict versioned artifact contracts, fail-closed validation, reducers, quality
review records, deterministic offline reports, and Linux/macOS clean-checkout
CI.

No fixture, healthy service, non-empty response, or generated report implies a
live provider passed, a benchmark score is valid, or a release is ready.

## Credential-free workflow

**A source checkout can run one complete fake benchmark path from manifest to
offline report.** Requirements are Python 3.11 or newer and
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

## Phase quality review

**Phase review is a separate post-report operation with its own bundle, plan,
approval, budget, occurrence, attempts, usage, resources, and cost evidence.**
Use `oamb phase --help` for the bundle, AI plan/run, local human confirmation,
gate validation, and acceptance-report commands. After the canonical AI review
passes, `oamb phase human-review record` requires one interactive TTY entry of
`STATUS BUNDLE_HASH AI_HASH NONCE` and writes one create-only version 2 human
record at `human-review.json` inside that review evidence directory. A second
record in the same authoritative review root is rejected. This local benchmark
confirmation uses no signing key or model client;
version 1 signed artifacts remain readable only as historical contracts.

The recorded-output reducer is explicitly offline. A model-backed AI review is
constructed only after the exact role, runtime, approval, budget, request
inventory, and artifact durability preflight close; normal tests use a fake
client and make no external call.

## Runtime contracts

**Persisted artifacts are parsed by strict, immutable, versioned Pydantic
models rather than checked-in generated schema files.** Every persisted model
retains explicit `schema_name` and `schema_version` fields. The registry rejects
unknown names and versions, while semantic validators enforce identities,
cross-artifact references, accounting completeness, and report eligibility.

## Production workflow boundary

**Provider configuration, live execution, eligible comparisons, and acceptance
remain explicit external operations, not installation claims.** Do not
translate the fake commands into a real provider run or reuse their artifact
root for live evidence. Every live call requires a separately approved scope,
immutable runtime binding, budget, unique artifact root, stop conditions, and
retained raw receipts.

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

Provider state is preserved. Normal tests never delete databases, run memory
conformance, or execute a paid evaluation.

## Project policies

**Security, dataset, and dependency policies are maintained as repository
documents.**

- [`SECURITY.md`](SECURITY.md) — vulnerability reporting and secret safety.
- [`DATASETS.md`](DATASETS.md) — dataset provenance and redistribution rules.
- [`THIRD_PARTY.md`](THIRD_PARTY.md) — direct dependency and license inventory.

Licensed under Apache-2.0. Dataset and third-party artifacts retain their own
licenses and are not relicensed by OAMB.
