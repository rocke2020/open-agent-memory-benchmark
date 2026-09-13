# PromptPack Notices

> **TL;DR:** OAMB distributes its own answer wrappers and small, attributed MIT-
> licensed prompt excerpts. Versioned runtime manifests bind every executed
> template byte; downloaded datasets and private user packs are not included.

The LongMemEval judge templates come from `src/evaluation/evaluate_qa.py` at
revision `9e0b455f4ef0e2ab8f2e582289761153549043fc`. The source file SHA-256 is
`ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251`.
Its complete MIT notice is in `LICENSE.LongMemEval`.

OAMB intentionally uses different prompt ownership at each evaluation stage.
Memory extraction uses a self-curated, benchmark-neutral session context that
identifies the model as the assistant, retains the session timestamp, and omits
the LongMemEval name. This framing is better suited to normal assistant memory
behavior because it supplies role and temporal context without revealing the
benchmark identity. Answer generation uses OAMB's own evidence-grounded prompt.
Only scoring preserves the original LongMemEval judge rubrics above, with exact
source attribution and byte-pinned templates.

The extraction context is workload metadata rather than a PromptPack template.
The original LongMemEval context is:

```text
LongMemEval session {session_id} at {timestamp}
```

OAMB uses this adapted context:

```text
Session {session_id} - you are the assistant for this conversation - took place at {timestamp}.
```

This is a semantic adaptation, not only a wording change. The original
LongMemEval form exposes the benchmark identity but supplies no conversational
role. The OAMB form instead tells the memory extractor how to interpret the
conversation, while preserving the session identifier and canonical timestamp.
Removing the benchmark name keeps extraction closer to an ordinary assistant-
memory use case. This adaptation affects extraction context only; it does not
modify the attributed LongMemEval scoring rubrics.

The five MemoryAgentBench query templates come from `utils/templates.py` at
revision `fe1735de8cf8b9908e1e3d3b5612afc815698062`. The source file SHA-256 is
`148c40d48d19f155ae845482c4417ba59cfa7ae4e194019509e023bd3a8755dd`.
Its complete MIT notice is in `LICENSE.MemoryAgentBench`.

MemoryAgentBench is deferred from v0.1.0. These attributed templates remain
preserved research assets and license evidence; they are not selected by the
current balanced LME-60 LongMemEval release plan.

The executable manifests remain owned by the corresponding workload modules, while the exact LongMemEval ingestion, answer, and judge template bytes live in `src/oamb/workloads/prompt_templates/longmemeval/` so humans can inspect them and source and installed-wheel execution use one byte source of truth. LongMemEval retrieval is generation-free and uses the question bytes as the typed provider-native query, so it has no textual prompt template. Public evidence may include the attributed judge templates. A user-supplied pack defaults to `redistribution_allowed: false`; public evidence then retains only names, byte counts, content hashes, rendered hashes, and variable-value hashes.
