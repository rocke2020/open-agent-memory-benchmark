# OAMB v0.1.0 Evaluation Results

This directory is the public, reproducible OAMB v0.1.0 snapshot for the balanced 60-question LongMemEval screen. It contains all 180 judged provider results, the frozen plans, the detailed comparison report, and the one content-addressed analysis cache entry that exactly matches this report.

| Provider | Answer accuracy | Answer-visible context tokens |
|---|---:|---:|
| Hindsight | **57/60 (95.0%)** | 954.4k total / 15.91k mean |
| Mem0 | 52/60 (86.7%) | **392.6k total / 6.54k mean** |
| OpenViking | 51/60 (85.0%) | 927.6k total / 15.46k mean |

Hindsight recorded the highest answer accuracy. Mem0's standout result is its low context use: 392.6k answer-visible context tokens, 59% fewer than Hindsight. This is a screening comparison, not a universal provider ranking.

Open the [hosted report](https://rocke2020.github.io/open-agent-memory-benchmark/eval_results/v0.1.0/comparison-20260913-133717-82378/report.html), use the [offline HTML](comparison-20260913-133717-82378/report.html), or inspect its [machine-readable data](comparison-20260913-133717-82378/report.json). The report analysis and the evaluation's extraction, answer, and judge roles used the configured `deepseek-flash` model (DeepSeek-V4.1-Flash) at their frozen effort settings.

## Rebuild the report

After following the repository Quick Start to create `.env` and download the pinned LongMemEval source, run from the repository root:

```bash
./run.sh --generate-report --result-dir=./eval_results/v0.1.0
```

The command reads only this versioned snapshot and the pinned dataset source. `case-manifest.json` preserves the exact IDs used by this historical evaluation; a new evaluation run uses the current manifest and new IDs. With the same model endpoint, model ID, and effort configuration, the report command reuses `report-analysis-cache/` and does not rerun the analysis model. It creates `comparison/`, or a timestamped successor if that directory already exists; it does not rerun any provider evaluation.

The authoritative portable 60-question result files are:

- `hindsight-20260911-57-of-60/results/hindsight.json`
- `mem0-lme60-20260913-oamb/results/mem0.json`
- `openviking-lme60-20260913-ov-oamb/results/openviking.json`

Mem0 and OpenViking were each assembled from two disjoint, independently validated 30-question halves. Hindsight is a preserved 60-question mixed-resume result. The four intermediate half-run result maps are intentionally excluded because the final provider files above contain the complete portable results.

Run `shasum -a 256 -c SHA256SUMS` from this directory to verify the saved evaluation evidence. Human-readable `README.md` files are intentionally excluded from the evidence manifest. The selected LongMemEval content is redistributed under the upstream MIT license; see [the included notice](LICENSES/LongMemEval-MIT.txt).
