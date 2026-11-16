# LME-500: paired saved-context results

Overview: OAM reaches **466/500 (93.2%)** with **MiniMax M3.1 Flash native adaptive** answers. The secondary **DeepSeek Flash max** arm reaches **463/500 (92.6%)**. Both arms use the same 500 question IDs, identical saved OAM fast-recall contexts, and a common DeepSeek Flash high-effort judge. These are judge-accepted answer rates.

[Open the offline question browser](report.html) to inspect every question, reference answer, complete injected context, answer and judgment side by side. The [manifest](manifest.json) identifies all compressed evidence shards. The [failure register](lme500-failure-review.md) links every rejected selected answer.

## 1. Results and selection

| Answer model and reasoning | Accepted | Rejected | Rate |
| --- | ---: | ---: | ---: |
| MiniMax M3.1 Flash, native adaptive | 466 | 34 | **93.2%** |
| DeepSeek Flash, max | 463 | 37 | **92.6%** |

The paired outcomes are 448 accepted by both, 18 accepted only for MiniMax, 15 accepted only for DeepSeek, and 19 rejected for both. There are 500 distinct questions, rather than 1,000 distinct questions.

MiniMax uses native-adaptive answers and DeepSeek uses max-effort answers for the same 500 questions. The saved DeepSeek selection includes 38 reused historical max receipts and 462 other max receipts. These saved-context measurements are not a fresh blinded first trial. Exact configured model strings and settings are retained in the evidence.

Both arms reuse saved retrieval evidence. The identical contexts total **5,740,548 `o200k_base` tokens**, averaging **11,481.096 tokens per question**. These are answer-visible context tokens, excluding answer instructions and questions. They are not indexing or generation usage.

## 2. Dataset, roles and implementation

| Setting | Recorded value |
| --- | --- |
| Dataset | LongMemEval, full 500-question selection |
| Upstream revision | `98d7416c24c778c2fee6e6f3006e7a073259d48f` |
| Dataset SHA-256 | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| Recall | OAM fast, non-generative, saved contexts shared by both answer models |
| Insights and Profile | Disabled |
| Historical extraction | Reported DeepSeek Flash with low reasoning; exact configured spelling and uniform indexing revision are unavailable in this projection |
| Primary answer | `minimax-m3.1-flash`, native adaptive thinking |
| Secondary answer | Saved DeepSeek Flash configured model strings, max reasoning |
| Common judge | Saved DeepSeek Flash configured model strings, high reasoning |
| Scoring | Original per-question yes/no verdict, unchanged gold and denominator |
| Recall implementation | `da89864ae1dbbb288157e86a7cdaf80b95abdf72` |

The saved recall source snapshots for the 100-question, 180-question and final 180-question groups have identical 116-file hash maps; all 116 match the stated revision. The remaining 40-question selection records that same revision. [Recall source hashes](recall-source-hashes.json) allow independent source comparison. Historical indexing reused existing memories, so this does not establish one uniform extraction/indexing revision or a new release-wide end-to-end run. The original configured model strings are preserved in each saved request.

## 3. Evidence and limitations

Each arm contains ten compressed JSON shards of 50 questions. Each record retains the question, gold, exact complete context and SHA-256, tokenizer count, saved retrieval timing, selected visible answer, strict official judgment, exact answer/judge request bodies, response model, usage, finish reason and original receipt hash. Provider reasoning text, credentials, endpoint addresses and private storage paths are omitted. The HTML embeds each shared context once and restores it for both cards; the JSON shards retain complete contexts independently.

All 1,000 selected source receipt byte hashes were checked against the retained selection before projection. Every selected answer and judge has HTTP 200, stop termination, nonempty visible output and the expected recorded reasoning setting. Both answer requests include the exact context whose hash is published. The verdict text agrees with each published boolean and score. Failed attempts and unrelated diagnostics do not enter this selected-answer total.

This report does not claim judge infallibility, a controlled fresh comparison, or causal superiority of one answer model. Saved judge verdicts can disagree with source-grounded human interpretations. Detailed private source reviews and context-noise assessments are not redistributed here; the public failure register provides the selected observable records and leaves causal attribution unassessed. No missing metric is represented as zero. No new model, retrieval, indexing or database operation was made to package this evidence.

LongMemEval-derived questions, reference answers and conversations retain the [upstream MIT notice](LICENSES/LongMemEval-MIT.txt). The contexts contain benchmark conversation text and OAM-derived memories, not a newly collected personal conversation dataset.

## 4. Rebuild and inspect

From the repository root, rebuild only the offline HTML:

```bash
PYTHONPATH=src uv run python scripts/build_historical_reports.py eval_results/lme500
uv run pytest -q tests/reporting/test_historical_report.py tests/test_published_oam_evidence.py
```

The builder reads saved local files and makes no provider calls. Open `report.html` in a current browser; it works without a server or network. Read the compressed JSON directly for full saved request and response projections. Original unprojected receipt hashes identify source artifacts; they are not hashes of the projected public JSON.
