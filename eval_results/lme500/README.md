# Open Agent Memory v0.0.1 — LME-500

## Overview

Open the [interactive offline report](report.html) to browse all 500 questions, full injected contexts, answers, golds, judgments and recorded retries. Its AMB-inspired layout supports category, text and outcome filters and separate first-pass / policy-final views. Download `report.html` and open it directly in a current browser; no setup or evaluation calls are needed. Full request/response archive links use the adjacent evidence files. See [rebuild instructions](../README.md).

**OAM achieves 94.0% accuracy (470/500)** on the 500 distinct questions in LongMemEval cleaned S, using the historical campaign's recorded single-retry policy. This report reuses the completed `lme-v261026` results and failure reviews. No ingestion, retrieval, answer generation, judging, or model-based review is rerun for this addition.

This is a historical evaluation of OAM v0.0.1 development snapshots, consolidated from six cohorts. It provides a useful performance reference without paying to repeat the campaign. It is not a fresh test of the current checkout or a newly matched comparison against another provider.

After this campaign, OAM added shared and dedicated approximate-nearest-neighbor index policies for large collections. The saved LME-500 scores were not rerun against that change; its effect on these results is unmeasured. The separate [BEAM-10M report](../beam10m/README.md) covers long histories under its own recorded configuration.

## Performance

| Metric | Recorded result | Meaning |
| --- | ---: | --- |
| Answer accuracy | **94.0% (470/500)** | Final official outcomes under the recorded retry policy; all 30 remaining failures stay in the denominator. |
| Question coverage | **500 unique IDs** | Six cohort selections cover the complete pinned cleaned-S dataset. Nine shared LME-12/LME-42 IDs are counted once. |
| Indexing generation tokens | **618,864,325** | Selected-history input plus output tokens; excludes embeddings and is not total campaign billing. |
| Answer-visible context tokens | **6,201,308** total; **12,402.62** mean | Exact saved answer-context counts using tiktoken 0.14.0, `o200k_base.encode_ordinary`; context size is separate from billed usage. |
| Mean retrieval latency | **Approximately 748.25 ms** | Weighted from reported cohort raw means with three samples per question; descriptive across different runtime conditions. |
| Full-campaign indexing duration | Unavailable | Resumed histories lack some original timing measurements; overlapping per-history durations cannot supply campaign wall time. |
| Total campaign cost | Unknown | Failed/abandoned scopes, uncertain external outcomes, evaluation overhead, and other usage are not completely captured. |

The saved accuracy is an outcome of the complete memory → answer → judge pipeline. Context size, indexing generation tokens, retrieval latency, and monetary cost measure different things and must not be combined into one score. Reasoning usage is already included in output tokens.

## Evaluation settings

| Setting | Historical campaign |
| --- | --- |
| Dataset | LongMemEval cleaned S, all 500 questions. |
| Dataset revision | `98d7416c24c778c2fee6e6f3006e7a073259d48f` |
| Dataset SHA-256 | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| OAM indexing and retrieval | Phase-one indexing; fast recall; L2 Insights and Profile disabled. |
| Models | DeepSeek Flash for extraction, answering, and judging; actual endpoint/configuration changes remain in the source cohort reports. |
| Embedding and tokenization | Qwen3 embedding with a 32K-token input limit; OAM used its approximate tokenizer during the evaluation. |
| Reasoning | Extraction low; original answer high; eligible single answer retry max; judge high. |
| Retry policy | A reviewed eligible failure may receive one max-effort answer on the unchanged original context and standard prompt, followed by high-effort judging. Clearly incorrect gold blocks retry; the official denominator remains unchanged. |

The v0.0.1 label identifies the development version, not one immutable release across every cohort. For example, the first LME-120 records product revision `1d8554b3e9fcaf8b169b7c542146dfd6e074f361`, whose [package metadata](https://github.com/rocke2020/open-agent-memory/blob/1d8554b3e9fcaf8b169b7c542146dfd6e074f361/pyproject.toml) declares version `0.0.1`. Preserve each cohort's runtime identity when using its results.

<details>
<summary>Supplementary: original-answer accuracy and model limitations</summary>

Before the permitted max-effort retries, the score was **90.6% (453/500)**. Seventeen outcomes subsequently changed to correct, producing the headline result. These are preserved outcomes from the same campaign, not an additional evaluation in OAMB.

DeepSeek Flash's answer capability is a relevant limitation when interpreting this pipeline score. The saved reviews contain concrete cases where OAM delivered the necessary evidence but the model still answered incorrectly. These support answer-level failure attribution; they do not measure the improvement a larger or stronger model would achieve. High and max use the same model, so the observed recoveries are not a comparison against a stronger model and do not isolate reasoning effort from answer/judge variability.

</details>

## Answer-model failures supported by the saved reviews

The following examples come from the existing [consolidated failure review](https://github.com/rocke2020/agent-memories/blob/54c3b12c3a9154d37e91ae95cb85b68335f222e1/eval/lme/lme-v261026/lme500-failure-review.md). They demonstrate that some judged failures arise in evidence use or answer interpretation even when the necessary source material reached the answer model.

| Case | Evidence available to the answer | Observed answer issue |
| --- | --- | --- |
| `gpt4_15e38248` — furniture count | All four source-supported actions, including the Casper mattress purchase, were visible. | Both answers excluded the mattress as bedding and counted three. The review supports an answer interpretation error, with no demonstrated omission of the required evidence. |
| `031748ae_abs` — initial engineering team | The actual Senior Software Engineer title, initial four engineers, later five, and a distinct manager were visible. | Both answers returned four while accepting the question's unsupported Software Engineer Manager premise, instead of qualifying or identifying insufficient information. |
| `09ba9854` — taxi versus train savings | The user's repeatedly affirmed personal fare quotes and conflicting generic assistant estimates were visible. | Both answers preferred the generic estimate over the personal quote; the max answer noticed the implied $50 saving but dismissed its basis. The review supports an evidence-authority/qualification issue. |

These findings coexist with OAM context-selection weaknesses, including unrelated and repetitive material. Evidence presence does not establish optimal retrieval or prove that context noise had no effect. Other reviewed failures involve ambiguous or incorrect gold answers; the original official scores remain unchanged. No exclusive cause or count of all answer-model failures is inferred from these examples.

## Cohort reconciliation and source

| Disjoint contribution | Questions | Final correct | Indexing generation tokens | Answer-context tokens |
| --- | ---: | ---: | ---: | ---: |
| Original pilot IDs outside LME-42 | 3 | 3 | 3,474,809 | 39,153 |
| LME-42, including nine reused pilot IDs | 42 | 40 | 52,497,277 | 521,504 |
| First LME-120 | 120 | 112 | 147,953,673 | 1,494,986 |
| LME-5 | 5 | 4 | 6,358,416 | 57,868 |
| Next LME-120 | 120 | 113 | 147,985,495 | 1,471,023 |
| Final LME-210 | 210 | 198 | 260,594,655 | 2,616,774 |
| **Total** | **500** | **470** | **618,864,325** | **6,201,308** |

The authoritative source is the [LME-500 summary](https://github.com/rocke2020/agent-memories/blob/54c3b12c3a9154d37e91ae95cb85b68335f222e1/eval/lme/lme-v261026/lme500-summary.md) and its linked cohort reports, selections, and consolidated failure review, pinned here to source repository revision `54c3b12c3a9154d37e91ae95cb85b68335f222e1`. OAMB reuses the reported measurements and reconciles selection coverage and aggregate arithmetic offline; it does not independently rescore raw answers or reconstruct complete billing.

Full-500 source-reviewed context-noise coverage is unavailable, so this report presents no aggregate noise percentage. The historical OAM result is separate from both the three-provider LME-60 comparison and new runs through the [native OAM workflow](../../docs/native-oam.md). It is not a native OAMB execution capsule or a controlled provider-performance win.

## Public per-question evidence

The [manifest](manifest.json) lists six compressed `oam/*.json.gz` shards covering all 500 distinct question IDs. They contain the original question and gold answer, the exact saved answer-visible context and its SHA-256, recorded context tokens and retrieval time, first-pass answer/judge messages and outputs, any recorded max-effort retry, and the policy-final answer and score. The three pilot-only questions and the LME-42 contribution are disjoint; the nine reused pilot IDs are counted only in LME-42.

`first_pass` and `retry` preserve both attempts. `policy_final_attempt` selects the retry whenever one was recorded, including any lower-scoring outcome; `answer`, `score` and `correct` expose that selected result. The saved LME judge output is a yes/no verdict, so no richer judging explanation is invented. Supplier usage included in an attempt is that receipt's measurement, not complete campaign billing.

Historical runtime revisions and the reported protocol are recorded in the manifest; unavailable revisions remain null. Exact saved LME answer and judge message bodies are included, while credentials, endpoints and private storage identities are omitted. This is a public projection of historical evidence, not a native execution capsule or a fresh run. LongMemEval content retains its [upstream MIT notice](LICENSES/LongMemEval-MIT.txt).

From the repository root, run `uv run pytest -q tests/test_published_oam_evidence.py` to recompute coverage, scores, policy selection and exact context-token counts from the public shards. Python's standard `gzip` and `json` modules can open each shard without OAM, a database or model credentials.
