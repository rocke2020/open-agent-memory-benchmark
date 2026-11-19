# Evaluation Results

Published OAMB results are grouped by evaluation workload. OAM evaluation is the primary workflow; the three-provider LME-60 screen is a separate comparison.

| Workload | Published content |
| --- | --- |
| [LongMemEval S, November 17](lme500-20261117/) | OAM 95.2% vs published Hindsight 94.6%, with an [offline report](lme500-20261117/report.html) |
| [BEAM-10M, November 19](beam10m-20261119/) | OAM 78.23% vs published Hindsight 64.08% mean rubric credit, with [200 paired outcomes](beam10m-20261119/scores.csv) |
| [Earlier LME-500](lme500/) | [Offline report](lme500/report.html), recorded settings and compressed per-question evidence |
| [Earlier BEAM-10M saved-context replay](beam10m/) | [Offline report](beam10m/report.html) and compressed per-question evidence |
| [LME-60](lme60/) | Complete saved Hindsight, Mem0 and OpenViking comparison snapshots and offline report |

Each workload retains its own run identities, evaluated product versions, model settings and provenance. Ongoing runs belong under ignored `outputs/`; downloaded datasets remain under ignored `datasets/`. Organizing these entries does not rerun an evaluation.

Download a linked `report.html` and open it directly in a browser. Each file embeds the available saved answer contexts, outcomes, references and judgments, with category, search and outcome filters. Links to receipt archives require the adjacent evidence files.

To rebuild the earlier HTML reports from the published shards without evaluation calls, run `PYTHONPATH=src uv run python scripts/build_historical_reports.py eval_results/lme500 eval_results/beam10m` from the repository root.
