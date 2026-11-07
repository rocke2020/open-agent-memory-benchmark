# Dataset Policy

Full downloaded benchmark payloads stay under ignored `datasets/`. Repository scripts fetch frozen LongMemEval S, BEAM-10M and deferred MemoryAgentBench inputs and reject bytes that do not match their pinned SHA-256 values. Versioned release evidence under `eval_results/` may include selected question content needed to audit a report when the upstream license permits redistribution.

MemoryAgentBench remains downloadable, pinned research input but is deferred from v0.1.0. The planned release comparison uses the balanced LME-60 selection from LongMemEval S: ten questions from each of six types, with the same ordered 60-question manifest for every provider. Retaining MAB bytes and provenance does not create a MAB cell or support claim.

OAMB does not treat benchmark data as project-owned source code. Every dataset binding must identify its authoritative source, exact revision, file hashes, license, redistribution policy, and any required local download procedure.

Raw, downloaded, restricted, or bulk datasets are not committed by default. Small generated or license-compatible fixtures and versioned evaluation evidence may be tracked when they are necessary for reproducibility and contain no restricted expression. The v0.1.0 LongMemEval evidence retains its upstream MIT notice.

Derived manifests preserve source row and question ordinals, repeated raw IDs, nested answer values, and provenance. They never modify, deduplicate, or relabel the immutable source dataset. External historical evidence remains external and cannot be converted into OAMB-native execution evidence.

The distributed historical evidence pack is a small factual projection, not a benchmark dataset or a native run capsule. It retains ordered case identifiers, categories, verdicts, producer-formatted context-token counts, retrieval time, and pinned source provenance while excluding questions, answers, contexts, reasoning, and raw responses. Its compatibility with OAMB-native runs remains `unknown`.

## Download benchmark inputs

The scripts resolve immutable Hugging Face revisions through the locked `download` dependency group, install files atomically, and refuse to overwrite an existing mismatched path.

Run from the repository root:

```bash
./scripts/download/longmemeval.sh
./scripts/download/beam10m.sh
./scripts/download/memoryagentbench.sh
```

The first command installs the LongMemEval S file at revision `98d7416c24c778c2fee6e6f3006e7a073259d48f` under `datasets/longmemeval-cleaned/`. The MemoryAgentBench command installs its four Parquet files plus the ReDial `entity2id.json` catalog at revision `7ea066982b140a19337e17e60d45d4076e042faf` under `datasets/MemoryAgentBench/`. Each directory contains `REVISION` and `SHA256SUMS` manifests generated only after every requested file passes its pinned checksum.

BEAM-10M comes from [Mohammadta/BEAM-10M](https://huggingface.co/datasets/Mohammadta/BEAM-10M) at revision `9b2096193fe74e2837e4713e483351e19817773c`. The helper downloads the two pinned `data/10M-*.parquet` shards and exports rows in shard order using AMB's standard-library JSON serialization. It verifies the resulting `datasets/beam/10m.json` against campaign SHA-256 `c16643a7cc7d93c1ed5509568cdd9e4c387dcd60414e68dde740b54064e68ba8`, preserving nested fields and original questions. The Parquet download is about 344 MB and the JSON export about 1.03 GB; both remain outside Git. The dataset retains its upstream license and is not redistributed with OAMB. Repeated downloads reuse verified output unchanged; a mismatched destination is preserved and rejected.
