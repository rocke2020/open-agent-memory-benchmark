# PromptPack Notices

## Overview

OAMB distributes answer templates and attributed LongMemEval and BEAM prompt excerpts. LongMemEval excerpts retain their MIT notice; the Mem0-origin BEAM excerpts retain their Apache-2.0 notice. Executed template bytes are bound in run evidence; downloaded datasets and private user packs are not included. The [native OAM workflows](../docs/native-oam.md) use the recorded campaign prompts described below; the separate LME-60 comparison uses the adapted extraction context and answer wrapper described next.

## LME-60 comparison prompts

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

The executable manifests remain owned by the corresponding workload modules, while the exact LongMemEval ingestion, answer, and judge template bytes live in `src/oamb/workloads/prompt_templates/longmemeval/` so humans can inspect them and source and installed-wheel execution use one byte source of truth. LongMemEval retrieval is generation-free and uses the question bytes as the typed provider-native query, so it has no textual prompt template. Public evidence may include the attributed judge templates. A user-supplied pack defaults to `redistribution_allowed: false`; public evidence then retains only names, byte counts, content hashes, rendered hashes, and variable-value hashes.

## Native OAM campaign prompts

The native OAM LME answer and judge template bytes match the recorded campaign's protocol files. BEAM's answer and rubric templates in `src/oamb/workloads/prompt_templates/beam/` reproduce the campaign's AMB revision `3d1514c75f7940064945edcd9865152fffb80d6f`, specifically `BEAMDataset.build_rag_prompt` and `_rubric_item_score` in [dataset/beam.py](https://github.com/vectorize-io/agent-memory-benchmark/blob/3d1514c75f7940064945edcd9865152fffb80d6f/src/memory_bench/dataset/beam.py). AMB attributes these prompt variants to Mem0's BEAM answer-generation and nugget-judge prompts; the upstream [Mem0 prompt source](https://github.com/mem0ai/memory-benchmarks/blob/4b61c5d31b9c668a12b4f5e78064248a02c82d2b/benchmarks/beam/prompts.py) is Apache-2.0, with its [license retained here](LICENSE.Mem0). These are the recorded campaign variants, not a switch to current upstream defaults.

The native request builder retains the recorded BEAM JSON system/schema message, `thinking: enabled`, and structured `{reasoning, answer}` response. Only its `answer` field is scored. Each rubric item is judged independently, numeric scores map to the nearest 0/0.5/1 bucket using the recorded thresholds, and the question score averages all rubric items. Event-ordering nuggets retain the campaign's mention framing. Invalid/missing model output is retained and stops execution rather than silently becoming a zero score. A separate source/context/gold review admits at most one max-effort answer on unchanged prompt and context, followed by high-effort judging.
