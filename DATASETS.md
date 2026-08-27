# Dataset Policy

> **TL;DR:** Downloaded benchmark payloads stay under ignored `datasets/`.
> Repository scripts fetch only the frozen LongMemEval S and MemoryAgentBench
> inputs and reject bytes that do not match their pinned SHA-256 values.

OAMB does not treat benchmark data as project-owned source code. Every dataset
binding must identify its authoritative source, exact revision, file hashes,
license, redistribution policy, and any required local download procedure.

Raw, downloaded, restricted, or bulk datasets are not committed by default.
Small generated or license-compatible fixtures may be tracked when they are
necessary for fail-capable tests and contain no restricted expression.

Derived manifests preserve source row and question ordinals, repeated raw IDs,
nested answer values, and provenance. They never modify, deduplicate, or
relabel the immutable source dataset. External historical evidence remains
external and cannot be converted into OAMB-native execution evidence.

## Download benchmark inputs

The two scripts resolve immutable Hugging Face revisions through the locked
`download` dependency group, install files atomically, and refuse to overwrite
an existing mismatched path.

Run from the repository root:

```bash
./scripts/download/longmemeval.sh
./scripts/download/memoryagentbench.sh
```

The first command installs the LongMemEval S file at revision
`98d7416c24c778c2fee6e6f3006e7a073259d48f` under
`datasets/longmemeval-cleaned/`. The second installs the four MemoryAgentBench
Parquet files plus the ReDial `entity2id.json` catalog at revision
`7ea066982b140a19337e17e60d45d4076e042faf` under
`datasets/MemoryAgentBench/`. Each directory contains `REVISION` and
`SHA256SUMS` manifests generated only after every requested file passes its
pinned checksum.
