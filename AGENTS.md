# AGENTS.md
This file provides context for AI coding assistants (Claude Code, Codex, etc.)

## Project scope

Open Agent Memory Benchmark (OAMB) is a clean-room Python package for
reproducible, evidence-first comparisons of self-hosted memory systems. The
default comparison reaches Hindsight, Mem0, and OpenViking through their REST
APIs. Mem0's optional Python SDK is a separate, non-default profile whose
evidence can never substitute for REST evidence.

Provider service preparation, offline implementation, live evidence, quality
acceptance, and release readiness are different completion states. Never infer
a live support or score claim from fixtures, schemas, a healthy container, or a
non-empty response.

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
  records, accounting shapes, behavior ports, and schema generation. It must
  not import concrete runtime, workload, adapter, model, or storage code.
- `src/oamb/artifacts/validation/` owns early structural rule registration and
  closed validation profiles. Semantic rules live with their later owners.
- `provider-services/` owns reproducible local API processes, immutable pins,
  isolated persistent state, and non-destructive service verification. It is
  beside the Python package, never imported by it.
- Future concrete adapters implement ports without importing runtime or one
  another. Only the CLI composition root selects concrete implementations.
- `schemas/` is generated from one explicit public-contract registry and is
  included in the built wheel as package data.

Avoid generic dumping grounds such as `utils.py`, `providers/`, or `services/`.
Names should state the property or boundary a module actually owns.

## Development workflow

Use Python 3.11+ and the committed `uv.lock`:

```bash
uv sync --locked --all-groups
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest tests/unit tests/contracts -q
python3 -m unittest discover -s provider-services/tests -p 'test_*.py' -v
uv run oamb schema check
uv build
```

Implement behavior test-first. Each validator rule needs a valid case and an
independent planted failure that proves the rejection branch can fire. Run
focused tests after each red-green cycle, then the affected offline suite.

Generated schemas are never hand-edited. Change the typed model, regenerate
with `uv run oamb schema build`, inspect the semantic diff, and rerun the drift
check. Required fields, identity inputs, state meanings, aggregation rules, or
parser behavior require a new schema/protocol version.

## Safety and evidence rules

- Never delete provider or database data. Tests must not perform real writes,
  real deletes, paid calls, or destructive container operations.
- Preserve partial, failed, interrupted, and unknown-outcome evidence. Never
  turn missing or unavailable measurements into zero.
- Default external allowance is zero. A real call requires an explicit scope,
  immutable runtime binding, approval, BudgetSpec, ceilings, and retained raw
  receipt.
- Keep execution state separate from derived validation disposition:
  `FINALIZED` does not mean `VALIDATED`, and `VALIDATED` does not mean a high
  score.
- Preserve provider-returned retrieval order. OAMB owns no reranker.
- Keep indexing usage and cost on the physical ingestion plan; do not multiply
  it by logical members or cases.
- Core import, CLI help, schema, and validation commands must not import provider
  SDKs, databases, model clients, or credentials. Optional dependencies load
  only inside the selected future factory.
- Never print secrets, resolved credential values, provider account keys, or
  private artifact content. Public validation issues contain codes and hashes,
  not offending values or host paths.

## Change discipline

Make the smallest change that closes the requested contract. Preserve unrelated
work and existing provider state. Use named constants for domain limits and one
canonical implementation for serialization, identity, transitions, schema
registration, and rule inventories.

Before claiming completion, run fail-capable checks against the final tree and
distinguish implemented from committed, fixture-tested from live-preflighted,
partial from complete, and committed from pushed or released.
