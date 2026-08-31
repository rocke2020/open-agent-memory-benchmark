# AGENTS.md
This file provides context for AI coding assistants (Claude Code, Codex, etc.)

## Project scope

Open Agent Memory Benchmark (OAMB) is a clean-room Python package for
reproducible, evidence-first comparisons of self-hosted memory systems. The
v0.1 comparison reaches Hindsight, Mem0, and the OpenViking session profile
through REST APIs on LongMemEval only. MemoryAgentBench is deferred. Mem0's
optional Python SDK is a separate, non-default profile whose evidence can never
substitute for REST evidence.

Provider service preparation, offline implementation, live evidence, quality
acceptance, and release readiness are different completion states. Never infer
a live support or score claim from fixtures, schemas, a healthy container, or a
non-empty response.

## Product promise and minimal dependencies

The product promise is:

> One local engineer can use a few clear commands to compare real memory
> providers on relevant datasets and produce a report that both an engineer and
> a manager can understand and audit.

Derive data flow from this promise and the minimum true dependency between
operations, not from existing function, module, phase, contract, or evidence
boundaries. For example, a question depends on its own history becoming ready;
it does not depend on every history or provider finishing ingestion. Release
work at the earliest correct boundary, and never add a global barrier merely to
simplify coordination or evidence sealing.

Keep the design simple and direct. Do not introduce unnecessary
over-engineering. Add an abstraction only when it removes existing duplication
or enforces a real user-visible requirement. Do not build speculative
compatibility families, control protocols, review layers, or benchmark-specific
architectures before the simple end-to-end product flow works with real data.

## Real-use vertical slice gate

The first hard delivery gate is the smallest safe real user flow, not broader
architecture or offline coverage. Before expanding contracts, schemas, control
records, review protocols, provider matrices, concurrency, or report
abstractions, complete this diagnostic flow with one provider and one
LongMemEval case:

```text
doctor/preflight
  → fresh isolated scope
  → chronological real ingestion
  → query-ready
  → real retrieval
  → answer
  → judge
  → freshly validated live capsule
  → offline HTML that opens for manual inspection
```

This one-case flow is a diagnostic gate, not a substitute for the complete T10
matrix. Until it passes, mock, fixture, contract, schema, and recorded-transport
tests prove only their own boundaries and do not unlock architecture expansion.
Only changes that directly unblock the next failing stage belong in the current
work; every other improvement goes to a deferred TODO. Do not optimize
parallelism or throughput before the same flow passes serially.

Apply stop-loss when one active working day produces no new real,
user-inspectable artifact, or when the diff grows materially while this gate
remains red: stop feature work, freeze scope, name the exact first failing
stage, preserve its evidence, and resume only with the smallest fix for that
stage. A design review, large passing test suite, or substantial diff is not
progress toward this gate without a new real artifact.

## Codes Architecture

Dependencies point inward toward strict contracts and behavior ports:

```text
contracts + ports
   ▲       ▲       ▲       ▲
workloads  memory systems  model clients  artifacts
   └──────────── injected into runtime ────────────┘
                         │
validated source roots ─┴─→ reporting
```

- `src/oamb/contracts/` owns canonical identities, states, immutable public
  records, accounting shapes, behavior ports, and initial-v1 artifact parsing. It must
  not import concrete runtime, workload, adapter, model, or storage code.
- `src/oamb/artifacts/validation/` owns early structural rule registration and
  closed validation profiles. Semantic rules live with their later owners.
- `src/oamb/external_evidence/` owns the read-only, hash-pinned historical
  evidence boundary. Its reports remain explicitly external and cannot acquire
  native capsule, attempt, indexing, billing, or compatibility claims.
- `src/oamb/reporting/` owns deterministic reducers, comparison projections,
  export validation, and offline HTML. It consumes validated capsules and does
  not own approval, signature, AI/human review, or phase-acceptance gates.
- `src/oamb/public_boundary.py` owns the fail-closed scan of tracked and
  distributed public artifacts for local-only paths and symlinks.
- `provider-services/` owns reproducible local API processes, immutable pins,
  isolated persistent state, and non-destructive service verification. It is
  beside the Python package, never imported by it.
- Concrete adapters implement ports without importing runtime or one another.
  Only an invoked execution command may lazily select a concrete
  implementation; core import and command help remain implementation-free.
- `src/oamb/contracts/schema.py` owns the fail-closed parser for one coherent
  initial-v1 contract family. Persisted `schema_name` and `schema_version`
  fields remain wire-format discriminators; unreleased historical forms and
  generated schema snapshots are not repository or package surfaces.

Avoid generic dumping grounds such as `utils.py`, `providers/`, or `services/`.
Names should state the property or boundary a module actually owns.

## Development workflow

Use Python 3.11+ and the committed `uv.lock`:

```bash
uv sync --locked --all-groups
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest -q
python3 -m unittest discover -s provider-services/tests -p 'test_*.py' -v
uv run python scripts/check_public_boundary.py
uv build
```

Implement behavior test-first. Each validator rule needs a valid case and an
independent planted failure that proves the rejection branch can fire. Run
focused tests after each red-green cycle, then the affected offline suite.

Before the first public release, change persisted contracts directly through
their typed initial-v1 models and parser, updating all producers, consumers,
fixtures, and validators together. Add a compatibility version only after
public, non-rerunnable evidence exists and users must reopen it.

## Safety and evidence rules

- Never delete provider or database data. Tests must not perform real writes,
  real deletes, paid calls, or destructive container operations.
- Preserve partial, failed, interrupted, and unknown-outcome evidence. Never
  turn missing or unavailable measurements into zero.
- Default external allowance is zero. A real call requires an explicit scope,
  immutable runtime binding, BudgetSpec, ceilings, and retained raw
  receipt.
- Keep execution state separate from derived validation disposition:
  `FINALIZED` does not mean `VALIDATED`, and `VALIDATED` does not mean a high
  score.
- Preserve provider-returned retrieval order. OAMB owns no reranker.
- v0.1.0 compares Hindsight, Mem0, and the OpenViking session profile on
  LongMemEval only. MemoryAgentBench is deferred and cannot satisfy a v0.1
  execution or release gate.
- Retrieval generation is disabled. Query embedding remains allowed, but
  Hindsight `reflect`, Mem0 generative reranking/search processing, OpenViking
  session-intent search, query planning/rewriting, and any other generative
  `memory_query` call are prohibited and must fail validation.
- A generative model binding includes configured alias, runtime-resolved model,
  effective thinking effort with provider-defined scale/rank, exact request or
  provider proof, and usage coverage. A model name alone is incomplete;
  embedding records thinking effort as not applicable.
- Keep indexing usage and cost on the physical ingestion plan; do not multiply
  it by logical members or questions.
- Core import, CLI help, artifact parsing, and validation commands must not import provider
  SDKs, databases, model clients, or credentials. Optional dependencies load
  only inside the selected future factory.
- Never print secrets, resolved credential values, provider account keys, or
  private artifact content. Public validation issues contain codes and hashes,
  not offending values or host paths.

## Change discipline

Make the smallest change that closes the requested contract. Preserve unrelated
work and existing provider state. Use named constants for domain limits and one
canonical implementation for serialization, identity, transitions, contract
registration, and rule inventories.

Before claiming completion, run fail-capable checks against the final tree and
distinguish implemented from committed, fixture-tested from live-preflighted,
partial from complete, and committed from pushed or released.
