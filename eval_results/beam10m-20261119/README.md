# BEAM-10M: OAM vs published Hindsight

## Overview

OAM exceeded Hindsight's published Agent Memory Benchmark (AMB) result in a comparison covering all 200 BEAM-10M questions. It earned **78.23% mean rubric credit and an 86.0% pass rate**, up **14.15 percentage points and 25 passes**, with **9.85% fewer saved context tokens**. OAM used low-cost **MiniMax M3.1 Flash** with adaptive thinking for both answering and judging. The benchmark tests abilities such as temporal reasoning, contradiction resolution and cross-session reasoning across ten conversations at the 10-million-token scale. [BEAM dataset and paper](https://huggingface.co/datasets/Mohammadta/BEAM-10M)

| BEAM-10M, 200 questions | OAM | Published Hindsight |
| --- | ---: | ---: |
| Mean rubric credit | **78.23%** | 64.08% |
| Pass rate (question score ≥ 0.5) | **86.0% (172/200)** | 73.5% (147/200) |
| Mean saved answer-context tokens (both recounted with `o200k_base`) | **24,676.88** | 27,373.25 |
| Answer model | MiniMax M3.1 Flash adaptive | Gemini 3.1 Pro Preview |
| Judge model | MiniMax M3.1 Flash adaptive | Gemini 2.5 Flash Lite |

OAM's independently recorded results exceed the top Hindsight entries for BEAM-10M and LongMemEval S in AMB's published results manifest checked November 19, 2026. [Pinned AMB results manifest](https://github.com/vectorize-io/agent-memory-benchmark/blob/c963dba91dfe6c459f64bb871c5dafb2f3de1bf1/results-manifest.json) · [LongMemEval S comparison](../lme500-20261117/README.md)

## 1. Matched evaluation protocol

Both results cover the same 200 question IDs, question text, reference answers, categories and rubrics. OAM used the **byte-identical answer and judge prompt templates** and scoring rules from AMB's archived BEAM evaluator. Replaying OAM's receipts matched all 200 answer and 660 accepted judge prompt texts byte for byte against that source with OAM's context and answers substituted. Three event-ordering judge calls added a JSON example in the MiniMax system wrapper to obtain a valid response. The archived judge text, schema and scoring rule stayed the same. The Hindsight archive contains its scores and saved contexts. Original Hindsight request bodies are unavailable. [Archived AMB evaluator](https://github.com/vectorize-io/agent-memory-benchmark/blob/5d5e8dbe56f140ad982ae61cbd73f43e100de706/src/memory_bench/dataset/beam.py) · [Published Hindsight artifact](https://github.com/vectorize-io/agent-memory-benchmark/blob/5d5e8dbe56f140ad982ae61cbd73f43e100de706/outputs/beam/hindsight/single-query/10m.json.gz)

Mean rubric credit averages the fractional question scores equally. The evaluator averages rubric-item judgments for most categories and scores event ordering with a normalized Kendall tau-b rank-agreement measure after matching event labels. Some archived answer prompts include reference-derived hints, such as event labels and summary aspects. Across the other 180 questions, OAM earned **76.48% and 152/180 passes**, versus **64.35% and 136/180** for Hindsight. Answer models, judge models and memory pipelines differ, so the measured gap describes end-to-end results.

## 2. Complex memory capabilities

Each capability includes 20 questions. The largest measured mean gains were in event ordering, temporal reasoning and information extraction. Cross-session reasoning remained the most difficult capability for both systems.

| Capability | OAM mean | Hindsight mean | OAM passes | Hindsight passes |
| --- | ---: | ---: | ---: | ---: |
| Event ordering | 94.05% | 61.63% | 20/20 | 11/20 |
| Temporal reasoning | 71.25% | 41.25% | 19/20 | 16/20 |
| Information extraction | 73.75% | 51.25% | 15/20 | 10/20 |
| Summarization | 96.75% | 78.21% | 20/20 | 18/20 |
| Contradiction resolution | 71.25% | 56.88% | 20/20 | 19/20 |
| Abstention | 90.00% | 80.00% | 18/20 | 16/20 |
| Knowledge update | 75.00% | 65.00% | 16/20 | 14/20 |
| Multi-session reasoning | 21.96% | 16.58% | 5/20 | 4/20 |
| Preference following | 97.08% | 97.50% | 20/20 | 20/20 |
| Instruction following | 91.25% | 92.50% | 19/20 | 19/20 |

## 3. Context and model usage

OAM reused existing L1 indexes and original conversation excerpts. The 180 current normal-recall captures for conversations 2–10 used a **24,000-token native context cap** and averaged **23,974.32 answer-context tokens**, including the benchmark's memory heading. The other 20 used previously saved L1 contexts from conversation 1 and averaged 30,999.95 tokens. All 200 count toward the displayed full-run mean. The seven pinned OAM retrieval files and base configuration match the linked revision. The capture used an eight-second normal recall deadline and L1-only eligibility. [OAM source revision](https://github.com/rocke2020/open-agent-memory/commit/b3e0bb767d829a232515f3c6570876643355a3ed)

The context comparison counts both saved texts with `o200k_base`. Hindsight's published count averages 27,341.99 with `cl100k_base`. These are counts of saved memory text. Complete model input, total bill and comparable indexing cost are unavailable.

MiniMax lists **USD 0.15 per million uncached input tokens** and **USD 0.60 per million output tokens**. OAM recorded 890 answer and judge attempts with 5,408,015 input tokens and 781,691 output tokens, including 689,186 reasoning tokens. That usage covers scoring, including retained failed attempts. It excludes context audits, embedding and indexing. Actual campaign billing is unknown. [Official MiniMax pricing](https://platform.minimax.io/docs/pricing/overview#llm)

## 4. Inspect the result

The [200 paired outcomes](scores.csv) contain question IDs, fractional scores and saved context counts. Average each score column for mean rubric credit and count scores of at least 0.5 for passes. This data uses the official [BEAM-10M dataset revision](https://huggingface.co/datasets/Mohammadta/BEAM-10M/tree/9b2096193fe74e2837e4713e483351e19817773c) with exported JSON SHA-256 `c16643a7cc7d93c1ed5509568cdd9e4c387dcd60414e68dde740b54064e68ba8`. The compressed Hindsight artifact has SHA-256 `594f4d8f1fe3158ea4f144fdf90be55c578505e698ef644d992a4c5c66b60fad`.

The earlier [DeepSeek Flash saved-context replay](../beam10m/README.md) used a different scoring setup and has its own offline report. The saved MiniMax scores here use the archived AMB protocol described above.

The current native BEAM workflow uses its own prompt and scoring profile. Re-scoring this campaign requires the archived evaluator, the recorded model settings and the exact saved contexts. [Native OAM workflow](../../docs/native-oam.md)
