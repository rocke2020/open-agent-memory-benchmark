# Open Agent Memory Benchmark

Open Agent Memory Benchmark (OAMB) is a clean-room, evidence-first framework
for comparing self-hosted agent-memory systems through reproducible API
boundaries. It keeps execution evidence, validation, metrics, resource usage,
cost, and reports separate so a partial run cannot be mistaken for a score.

The current repository contains:

- exact-pinned REST service preparation for Hindsight, Mem0, and OpenViking;
- the dependency-free OAMB contract and behavior-port kernel;
- deterministic JSON Schema generation and drift checking;
- fail-closed early structural validation with planted failures;
- a locked Python package and credential-free Linux/macOS CI gate.

No live benchmark, model-readiness probe, memory-conformance write, score, or
release claim is implied by those components.

## Quick start

Requirements: Python 3.11 or newer and
[`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --locked --all-groups
uv run oamb --help
uv run pytest tests/unit tests/contracts -q
uv run oamb schema check
uv build
```

These commands are offline after dependencies are available. They do not read
provider credentials or start provider services.

## Contract kernel

Public records are strict, immutable Pydantic models with explicit
`schema_name` and `schema_version`. `src/oamb/contracts/schema.py` is the only
public-contract registry. Schemas under `schemas/` are generated artifacts;
never edit them by hand.

```bash
uv run oamb schema build
uv run oamb schema check
```

The early validator covers schema/version, identity, references, counts,
hashes, state transitions, and configured-value exposure. Semantic workload,
adapter, accounting, comparison, and export validation arrive with their owning
implementation layers.

## Provider API services

The default comparison boundary uses REST for all three systems. The optional
Mem0 Python SDK profile is separate and cannot substitute for REST evidence.
See [`provider-services/README.md`](provider-services/README.md) for the
non-destructive service-only workflow.

Start them explicitly with:

```bash
cd provider-services && ./bin/provider-services up
```

Provider state is preserved. Normal tests never delete databases, run memory
conformance, or execute a paid evaluation.

## Project policies

- [`SECURITY.md`](SECURITY.md) — vulnerability reporting and secret safety.
- [`DATASETS.md`](DATASETS.md) — dataset provenance and redistribution rules.
- [`THIRD_PARTY.md`](THIRD_PARTY.md) — direct dependency and license inventory.
- [`AGENTS.md`](AGENTS.md) — contributor and coding-agent conventions.

Licensed under Apache-2.0. Dataset and third-party artifacts retain their own
licenses and are not relicensed by OAMB.
