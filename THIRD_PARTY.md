# Third-Party Components

This inventory covers direct dependencies introduced by the package foundation.
Transitive dependency details and exact versions are frozen in `uv.lock`.

| Component | Role | License | Distribution |
|---|---|---|---|
| Pydantic | Strict immutable runtime contracts and versioned artifact parsing | MIT | Runtime dependency |
| Typer | CLI composition | MIT | Runtime dependency |
| HTTPX | Original REST adapters and OpenAI-compatible model transport | BSD-3-Clause | Runtime dependency |
| tiktoken | Protocol-frozen ordinary-text token counting and chunking | MIT | Runtime dependency |
| Apache Arrow | Schema-preserving reads of pinned Parquet workload inputs | Apache-2.0 | Runtime dependency |
| Hatchling | Python wheel and source build backend | MIT | Build dependency |
| pytest | Unit and contract test runner | MIT | Development dependency |
| Hypothesis | Canonical identity and state property tests | MPL-2.0 | Development dependency |
| Ruff | Formatting and static linting | MIT | Development dependency |
| mypy | Static type checking | MIT | Development dependency |
| pytest-asyncio | Async model-transport tests | Apache-2.0 | Development dependency |
| huggingface_hub | Pinned benchmark dataset downloads | Apache-2.0 | Download dependency |
| HTTPX with SOCKS support | Hub transport through configured proxies | BSD-3-Clause | Download dependency |
| LongMemEval prompt excerpts | Attributed judge PromptPack templates | MIT | Embedded runtime PromptPack data |
| MemoryAgentBench prompt excerpts | Attributed MAB-65 answer PromptPack templates | MIT | Embedded runtime PromptPack data |

Provider images, source archives, datasets, PromptPacks, and imported evidence
retain their own upstream licenses and notices. Their inclusion in a local
workflow does not relicense them under Apache-2.0.

The complete notices for distributed prompt excerpts are installed from
`prompt-packs/`. Prompt manifests bind the exact upstream revision, source-file
SHA-256, extracted-template SHA-256, and OAMB-rendered template SHA-256.
