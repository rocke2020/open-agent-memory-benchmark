# Evaluation Results

Published OAMB results are grouped by evaluation workload. OAM evaluation is the primary workflow; the three-provider LME-60 screen is a separate comparison.

| Workload | Published content |
| --- | --- |
| [LME-500](lme500/) | [Offline HTML report](lme500/report.html), recorded settings and compressed per-question evidence |
| [BEAM-10M](beam10m/) | [Offline HTML report](beam10m/report.html), OAM / Hindsight saved-context replay and compressed per-question evidence |
| [LME-60](lme60/) | Complete saved Hindsight, Mem0 and OpenViking comparison snapshots and offline report |

Each workload retains its own run identities, evaluated product versions, model settings and provenance. Ongoing runs belong under ignored `outputs/`; downloaded datasets remain under ignored `datasets/`. Organizing these entries does not rerun an evaluation.

Download either `report.html` and open it directly in a current browser. Each file embeds the complete saved answer contexts and question outcomes; browsing needs no server, credentials or network. The AMB-inspired interface includes category/search/outcome filters, first-pass and policy-final views, answers, gold/rubrics, judgments and retry history. Links to full request/response archives require the adjacent evidence files. GitHub's source view does not execute the report.

To rebuild both HTML files from the published shards without evaluation calls, run `uv run python scripts/build_historical_reports.py eval_results/lme500 eval_results/beam10m` from the repository root.
