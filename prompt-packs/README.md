# PromptPack Notices

> **TL;DR:** OAMB distributes its own answer wrappers and small, attributed MIT-
> licensed prompt excerpts. Versioned runtime manifests bind every executed
> template byte; downloaded datasets and private user packs are not included.

The LongMemEval judge templates come from `src/evaluation/evaluate_qa.py` at
revision `9e0b455f4ef0e2ab8f2e582289761153549043fc`. The source file SHA-256 is
`ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251`.
Its complete MIT notice is in `LICENSE.LongMemEval`.

The five MemoryAgentBench query templates come from `utils/templates.py` at
revision `fe1735de8cf8b9908e1e3d3b5612afc815698062`. The source file SHA-256 is
`148c40d48d19f155ae845482c4417ba59cfa7ae4e194019509e023bd3a8755dd`.
Its complete MIT notice is in `LICENSE.MemoryAgentBench`.

MemoryAgentBench is deferred from v0.1.0. These attributed templates remain
preserved research assets and license evidence; they are not selected by the
current LongMemEval-only release plan.

The executable manifests and exact template bytes live with the corresponding
workload modules so source and installed-wheel execution use one byte source of
truth. Public evidence may include these attributed templates. A user-supplied
pack defaults to `redistribution_allowed: false`; public evidence then retains
only names, byte counts, content hashes, rendered hashes, and variable-value
hashes.
