# Open Agent Memory Benchmark

> **TL;DR:** Open Agent Memory Benchmark (OAMB) keeps execution evidence,
> validation, metrics, resource usage, cost, and reports separate so a partial
> run cannot be mistaken for a score. The repository currently offers a complete
> credential-free fake workflow; production provider composition is not yet a
> supported command-line workflow.

## Current status

**The offline evidence pipeline is implemented and testable without credentials,
while live benchmark execution remains gated.** The repository contains
exact-pinned REST service preparation for Hindsight, Mem0, and OpenViking,
strict versioned artifact contracts, fail-closed validation, reducers, quality
review records, and deterministic offline reports.

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

## Runtime contracts

**Persisted artifacts are parsed by strict, immutable, versioned Pydantic
models rather than checked-in generated schema files.** Every persisted model
retains explicit `schema_name` and `schema_version` fields. The registry rejects
unknown names and versions, while semantic validators enforce identities,
cross-artifact references, accounting completeness, and report eligibility.

## Production workflow boundary

**Provider configuration, live execution, eligible comparisons, and the final
user-facing production workflow remain pending distribution and CLI work.** Do
not translate the fake commands into a real provider run or reuse their artifact
root for live evidence. Live calls require separately approved scope, immutable
runtime bindings, budgets, and retained receipts.

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
